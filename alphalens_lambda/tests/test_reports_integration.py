from fastapi.testclient import TestClient
import pytest

import alphalens_lambda.app as backend
from alphalens_lambda.reports.config import ReportSettings
from alphalens_lambda.reports.formatting import build_sections_text, normalize_html
from alphalens_lambda.reports.graph import normalize_request, report_graph
from alphalens_lambda.reports.schemas import ReportSection
from alphalens_lambda.reports.services import (
    CerebrasReportService,
    GmailReportService,
    ReportDataService,
)


REPORT_PAYLOAD = {
    "type": "reports",
    "mode": "run",
    "job_id": "11111111-1111-4111-8111-111111111111",
    "question": "Prepare a EUR/USD weekly outlook ahead of US CPI",
    "instrument": "EUR/USD",
    "timeframe": "1D",
    "email": "client@example.com",
    "user_id": "22222222-2222-4222-8222-222222222222",
    "authenticated_user_id": "22222222-2222-4222-8222-222222222222",
    "sections": [
        {
            "id": "technical_analysis",
            "title": "Technical Analysis",
            "description": "Key levels and technical setup",
            "order": 2,
            "userNotes": "",
        },
        {
            "id": "market_overview",
            "title": "Market Overview",
            "description": "General market context",
            "order": 1,
            "userNotes": "Focus on Fed/ECB divergence",
        },
    ],
}


class FakeGraph:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def ainvoke(self, state):
        self.calls.append(state)
        return self.result


def test_report_request_accepts_direct_and_wrapped_payloads():
    direct = normalize_request({"raw_request": REPORT_PAYLOAD})
    wrapped = normalize_request({"raw_request": {"body": REPORT_PAYLOAD}})

    assert direct["question"] == REPORT_PAYLOAD["question"]
    assert wrapped["job_id"] == REPORT_PAYLOAD["job_id"]


def test_sections_match_n8n_sorting_and_format():
    sections = [ReportSection.model_validate(item) for item in REPORT_PAYLOAD["sections"]]

    assert build_sections_text(sections) == (
        "The user requested the following report sections:\n\n"
        "Section: Market Overview\n"
        "Description: General market context\n"
        "User Notes: Focus on Fed/ECB divergence\n\n"
        "Section: Technical Analysis\n"
        "Description: Key levels and technical setup"
    )


def test_html_contract_is_strict():
    result = normalize_html("```html\n<html><body><h2>Report</h2></body></html>\n```")

    assert result.startswith("<html><body>")
    assert result.endswith("</body></html>")


def test_report_route_does_not_call_existing_graphs(monkeypatch):
    report_response = {
        "job_id": REPORT_PAYLOAD["job_id"],
        "message": "SENT",
        "output": {"base_report": "<html><body>Report</body></html>"},
        "gmail_status": "SENT",
    }
    fake_report = FakeGraph({"response": report_response})
    monkeypatch.setattr(backend, "report_graph", fake_report)
    monkeypatch.setattr(
        backend,
        "build_run_graph",
        lambda: (_ for _ in ()).throw(AssertionError("Macro graph was called")),
    )
    monkeypatch.setattr(
        backend,
        "build_trade2_graph",
        lambda: (_ for _ in ()).throw(AssertionError("Trade graph was called")),
    )

    response = TestClient(backend.app).post("/run", json=REPORT_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == report_response
    assert fake_report.calls == [{"raw_request": REPORT_PAYLOAD}]


def test_wrapped_report_route_is_detected(monkeypatch):
    report_response = {
        "job_id": REPORT_PAYLOAD["job_id"],
        "message": "SENT",
        "output": {"base_report": "<html><body>Report</body></html>"},
        "gmail_status": "SENT",
    }
    fake_report = FakeGraph({"response": report_response})
    monkeypatch.setattr(backend, "report_graph", fake_report)

    response = TestClient(backend.app).post("/run", json={"body": REPORT_PAYLOAD})

    assert response.status_code == 200
    assert response.json() == report_response
    assert fake_report.calls == [{"raw_request": {"body": REPORT_PAYLOAD}}]


def test_report_route_returns_real_http_error(monkeypatch):
    class FailingGraph:
        async def ainvoke(self, state):
            raise RuntimeError("Report provider failed")

    monkeypatch.setattr(backend, "report_graph", FailingGraph())

    response = TestClient(backend.app).post("/run", json=REPORT_PAYLOAD)

    assert response.status_code == 500
    assert response.json() == {
        "job_id": REPORT_PAYLOAD["job_id"],
        "message": "ERROR",
        "error": "Report provider failed",
    }


def test_trade_and_macro_routing_contracts_are_unchanged(monkeypatch):
    fake_trade = FakeGraph({"trade_generation_output": {"result": "trade"}})
    fake_macro = FakeGraph({"output": {"final_answer": '{"result":"macro"}'}})
    fake_report = FakeGraph({"response": {"message": "SENT"}})
    monkeypatch.setattr(backend, "build_trade2_graph", lambda: fake_trade)
    monkeypatch.setattr(backend, "build_run_graph", lambda: fake_macro)
    monkeypatch.setattr(backend, "report_graph", fake_report)
    monkeypatch.setattr(backend, "supabase_update_job_status", lambda *args, **kwargs: {})

    client = TestClient(backend.app)
    trade_response = client.post(
        "/run",
        json={"type": "RAG", "mode": "trade_generation", "question": "Trade EUR/USD"},
    )
    macro_response = client.post(
        "/run",
        json={"type": "RAG", "mode": "run", "question": "Analyze EUR/USD"},
    )

    assert trade_response.status_code == 200
    assert macro_response.status_code == 200
    assert len(fake_trade.calls) == 1
    assert len(fake_macro.calls) == 1
    assert fake_report.calls == []


@pytest.mark.asyncio
async def test_report_graph_uses_cerebras_data_and_sends_gmail(monkeypatch):
    captured = {}

    async def fake_news(self):
        return {"articles": [{"headline": "Fed update", "summary": "Rates", "date": "2026-06-11"}]}

    async def fake_market(self, instrument, timeframe):
        return {"instrument": instrument, "timeframe": timeframe, "market_data": {"close": "1.1"}}

    async def fake_base(self, **kwargs):
        return {
            "message": {
                "content": {
                    "content": "Market Overview\nVerified base report.",
                    "citations": ["Fed update"],
                }
            }
        }

    async def fake_enrich(self, base_report_text, market_context):
        return f"{base_report_text}\nVerified close: {market_context['market_data']['close']}"

    async def fake_html(self, report_text):
        return "<html><body><h2>Market Overview</h2><p>Verified report.</p></body></html>"

    async def fake_send(self, recipient, html_body):
        captured.update(recipient=recipient, html=html_body)
        return {"id": "gmail-id"}

    monkeypatch.setattr(ReportDataService, "fetch_news", fake_news)
    monkeypatch.setattr(ReportDataService, "fetch_market_context", fake_market)
    monkeypatch.setattr(CerebrasReportService, "generate_base_report", fake_base)
    monkeypatch.setattr(CerebrasReportService, "enrich_report", fake_enrich)
    monkeypatch.setattr(CerebrasReportService, "build_html", fake_html)
    monkeypatch.setattr(GmailReportService, "send_html", fake_send)

    result = await report_graph.ainvoke({"raw_request": REPORT_PAYLOAD})
    response = result["response"]

    assert response["message"] == "SENT"
    assert response["gmail_status"] == "SENT"
    assert response["output"]["base_report"].startswith("<html><body>")
    assert captured["recipient"] == "client@example.com"


def test_reports_use_cerebras_model_by_default():
    settings = ReportSettings(
        cerebras_api_key="test",
        finnhub_token="test",
        twelve_data_api_key="test",
        gmail_client_id="test",
        gmail_client_secret="test",
        gmail_refresh_token="test",
        default_report_email="dorian.labry@gmail.com",
        cerebras_model="gpt-oss-120b",
        request_timeout_seconds=30,
    )

    assert CerebrasReportService(settings).settings.cerebras_model == "gpt-oss-120b"
    assert GmailReportService(settings).resolve_recipient(None) == (
        "dorian.labry@gmail.com"
    )


@pytest.mark.asyncio
async def test_cerebras_report_uses_verified_news_as_citation_fallback(monkeypatch):
    settings = ReportSettings(
        cerebras_api_key="test",
        finnhub_token="test",
        twelve_data_api_key="test",
        gmail_client_id="test",
        gmail_client_secret="test",
        gmail_refresh_token="test",
        default_report_email="dorian.labry@gmail.com",
        cerebras_model="gpt-oss-120b",
        request_timeout_seconds=30,
    )
    service = CerebrasReportService(settings)

    async def fake_complete(system_prompt, user_prompt):
        return '{"message":{"content":{"content":"Verified report","citations":[]}}}'

    monkeypatch.setattr(service, "complete", fake_complete)
    result = await service.generate_base_report(
        question="Outlook",
        sections_text="Sections",
        news_context={
            "articles": [
                {
                    "headline": "Fed update",
                    "source": "Test Wire",
                    "date": "2026-06-11",
                }
            ]
        },
        instrument="EUR/USD",
        timeframe="1D",
        custom_notes="",
    )

    assert result["message"]["content"]["citations"] == [
        {
            "headline": "Fed update",
            "source": "Test Wire",
            "date": "2026-06-11",
        }
    ]
