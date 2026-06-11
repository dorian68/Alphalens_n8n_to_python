import asyncio
import base64
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import json
from typing import Any

import httpx
from cerebras.cloud.sdk import Cerebras

from .config import ReportSettings
from .formatting import (
    extract_report_envelope,
    fallback_html,
    normalize_html,
)


RELEVANT_TERMS = (
    "inflation", "cpi", "gdp", "jobs", "payroll", "economy", "economic",
    "yield", "rates", "recession", "growth", "federal reserve", "fed", "ecb",
    "boj", "bank of england", "central bank", "monetary policy", "forex",
    "currency", "dollar", "euro", "yen", "sterling", "fx", "oil", "gold",
    "silver", "copper", "commodity", "commodities", "opec", "bitcoin",
    "ethereum", "crypto", "bonds", "treasury", "markets", "investors",
)


class ReportDataService:
    FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"
    FINNHUB_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/economic"
    TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, settings: ReportSettings):
        self.settings = settings

    async def fetch_news(self) -> dict[str, Any]:
        if not self.settings.finnhub_token:
            raise ValueError("FINNHUB_TOKEN is required for reports")
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.get(
                self.FINNHUB_NEWS_URL,
                params={"category": "general", "token": self.settings.finnhub_token},
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Finnhub news returned an unsupported response shape")

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=30)
        articles: list[dict[str, Any]] = []
        for article in data:
            timestamp = article.get("datetime")
            if not isinstance(timestamp, (int, float)):
                continue
            published_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            if published_at < cutoff or published_at > now:
                continue
            headline = str(article.get("headline") or "").strip()
            summary = str(article.get("summary") or "").strip()
            if not any(term in f"{headline} {summary}".lower() for term in RELEVANT_TERMS):
                continue
            articles.append(
                {
                    "headline": headline,
                    "summary": summary,
                    "datetime": timestamp,
                    "source": str(article.get("source") or "").strip(),
                    "date": published_at.date().isoformat(),
                }
            )
            if len(articles) >= 30:
                break
        return {"articles": articles}

    async def fetch_market_context(
        self, instrument: str, timeframe: str
    ) -> dict[str, Any]:
        interval = {
            "1H": "1h",
            "4H": "4h",
            "1D": "1day",
            "1W": "1week",
        }.get(timeframe.upper(), "1day")
        today = datetime.now(timezone.utc).date()
        calendar_from = today.isoformat()
        calendar_to = (today + timedelta(days=14)).isoformat()

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            market_request = client.get(
                self.TWELVE_DATA_URL,
                params={
                    "symbol": instrument,
                    "interval": interval,
                    "outputsize": 20,
                    "apikey": self.settings.twelve_data_api_key,
                },
            )
            calendar_request = client.get(
                self.FINNHUB_CALENDAR_URL,
                params={
                    "from": calendar_from,
                    "to": calendar_to,
                    "token": self.settings.finnhub_token,
                },
            )
            market_response, calendar_response = await asyncio.gather(
                market_request, calendar_request, return_exceptions=True
            )

        def parse_response(response: Any) -> Any:
            if isinstance(response, Exception) or response.status_code >= 400:
                return "Unavailable"
            try:
                return response.json()
            except Exception:
                return "Unavailable"

        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "market_data": (
                parse_response(market_response)
                if self.settings.twelve_data_api_key
                else "Unavailable"
            ),
            "economic_calendar": (
                parse_response(calendar_response)
                if self.settings.finnhub_token
                else "Unavailable"
            ),
        }


class CerebrasReportService:
    def __init__(self, settings: ReportSettings):
        if not settings.cerebras_api_key:
            raise ValueError("CEREBRAS_API_KEY is required for reports")
        self.settings = settings

    def _complete_sync(self, system_prompt: str, user_prompt: str) -> str:
        client = Cerebras(api_key=self.settings.cerebras_api_key)
        response = client.chat.completions.create(
            model=self.settings.cerebras_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            top_p=0.95,
            max_completion_tokens=16000,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Cerebras returned an empty response")
        return content.strip()

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(self._complete_sync, system_prompt, user_prompt)

    async def generate_base_report(
        self,
        question: str,
        sections_text: str,
        news_context: dict[str, Any],
        instrument: str,
        timeframe: str,
        custom_notes: str,
    ) -> dict[str, Any]:
        system_prompt = """You are a senior institutional FX and macro strategist.
Use only the supplied recent news. Never invent prices, facts, targets, trade
entries, or citations. Integrate every requested section and its user notes.
If the instrument is not FX, adapt the reasoning accordingly.
Return only valid JSON in exactly this envelope:
{"message":{"content":{"content":"final base report text","citations":[]}}}"""
        user_prompt = f"""Question: {question}
Instrument: {instrument}
Timeframe: {timeframe}
Additional custom notes: {custom_notes or "None"}

Requested sections:
{sections_text}

Verified recent news:
{json.dumps(news_context, ensure_ascii=False)}"""
        raw = await self.complete(system_prompt, user_prompt)
        report = extract_report_envelope(raw)
        citations = report["message"]["content"].get("citations", [])
        if not citations:
            report["message"]["content"]["citations"] = [
                {
                    "headline": article.get("headline", ""),
                    "source": article.get("source", ""),
                    "date": article.get("date", ""),
                }
                for article in news_context.get("articles", [])[:20]
            ]
        return report

    async def enrich_report(
        self, base_report_text: str, market_context: dict[str, Any]
    ) -> str:
        system_prompt = """Enrich the supplied institutional report using only the
verified market data and economic calendar supplied. Preserve all section names,
order, and structure. Never invent numbers. If data is unavailable, omit the
unsupported claim rather than fabricating it. Return only the complete enriched
report as plain text."""
        return await self.complete(
            system_prompt,
            f"""Base report:
{base_report_text}

Verified data:
{json.dumps(market_context, ensure_ascii=False, default=str)}""",
        )

    async def build_html(self, report_text: str) -> str:
        system_prompt = """Convert the supplied report into a professional Gmail
HTML email. Use inline CSS only. Convert titles to h2/h3 and bullets to ul/li.
Output raw HTML only. The first characters must be exactly <html><body> and the
last characters must be exactly </body></html>."""
        raw = await self.complete(system_prompt, report_text)
        try:
            return normalize_html(raw)
        except ValueError:
            return fallback_html(report_text)


class GmailReportService:
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    def __init__(self, settings: ReportSettings):
        self.settings = settings

    def resolve_recipient(self, email: str | None) -> str:
        return email or self.settings.default_report_email

    async def send_html(self, recipient: str | None, html_body: str) -> dict[str, Any]:
        required = {
            "GMAIL_CLIENT_ID": self.settings.gmail_client_id,
            "GMAIL_CLIENT_SECRET": self.settings.gmail_client_secret,
            "GMAIL_REFRESH_TOKEN": self.settings.gmail_refresh_token,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing Gmail configuration: {', '.join(missing)}")

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            token_response = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.settings.gmail_client_id,
                    "client_secret": self.settings.gmail_client_secret,
                    "refresh_token": self.settings.gmail_refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]

            message = EmailMessage()
            message["To"] = self.resolve_recipient(recipient)
            message["Subject"] = "[TEST-REPORT]"
            message.set_content("This report requires an HTML-capable email client.")
            message.add_alternative(html_body, subtype="html")
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
            response = await client.post(
                self.SEND_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                json={"raw": raw},
            )
        response.raise_for_status()
        return response.json()
