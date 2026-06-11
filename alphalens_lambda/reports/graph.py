from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import get_report_settings
from .formatting import (
    build_sections_text as format_sections_text,
    json_to_readable_string,
)
from .schemas import ReportRequest, ReportSection
from .services import (
    CerebrasReportService,
    ReportDataService,
    ResendReportService,
)


class ReportState(TypedDict, total=False):
    raw_request: dict[str, Any]
    body: dict[str, Any]
    request: ReportRequest
    question: str
    sections: list[ReportSection]
    email: str | None
    job_id: str | None
    instrument: str
    timeframe: str
    custom_notes: str
    sections_text: str
    news_context: dict[str, Any]
    readable_news: str
    market_context: dict[str, Any]
    base_report: dict[str, Any]
    enriched_report_text: str
    message: dict[str, Any]
    email_html: str
    email_status: str
    gmail_status: str
    response: dict[str, Any]


def normalize_request(state: ReportState) -> dict[str, Any]:
    raw_request = state.get("raw_request")
    if not isinstance(raw_request, dict):
        raise ValueError("Request body must be a JSON object")
    body = raw_request.get("body") if isinstance(raw_request.get("body"), dict) else raw_request
    request = ReportRequest.model_validate(body)
    if "reports" not in request.type.lower():
        raise ValueError("Unsupported request type")
    return {
        "body": body,
        "request": request,
        "question": request.question,
        "sections": request.sections,
        "email": request.email,
        "job_id": str(request.job_id) if request.job_id else None,
        "instrument": request.instrument or "Multi-Asset",
        "timeframe": request.timeframe or "1D",
        "custom_notes": request.custom_notes or "",
    }


def build_sections_text(state: ReportState) -> dict[str, str]:
    return {"sections_text": format_sections_text(state.get("sections", []))}


async def fetch_finnhub_news(state: ReportState) -> dict[str, Any]:
    news = await ReportDataService(get_report_settings()).fetch_news()
    return {"news_context": news, "readable_news": json_to_readable_string(news)}


async def fetch_market_context(state: ReportState) -> dict[str, Any]:
    context = await ReportDataService(get_report_settings()).fetch_market_context(
        state["instrument"], state["timeframe"]
    )
    return {"market_context": context}


async def generate_base_report(state: ReportState) -> dict[str, Any]:
    report = await CerebrasReportService(get_report_settings()).generate_base_report(
        question=state["question"],
        sections_text=state["sections_text"],
        news_context=state["news_context"],
        instrument=state["instrument"],
        timeframe=state["timeframe"],
        custom_notes=state["custom_notes"],
    )
    return {"base_report": report}


async def enrich_report(state: ReportState) -> dict[str, str]:
    base_text = state["base_report"]["message"]["content"]["content"]
    enriched = await CerebrasReportService(get_report_settings()).enrich_report(
        base_text, state["market_context"]
    )
    return {"enriched_report_text": enriched}


def normalize_report_envelope(state: ReportState) -> dict[str, Any]:
    citations = state["base_report"].get("message", {}).get("content", {}).get(
        "citations", []
    )
    return {
        "message": {
            "content": {
                "content": state["enriched_report_text"],
                "citations": citations if isinstance(citations, list) else [],
            }
        }
    }


async def build_html_email(state: ReportState) -> dict[str, str]:
    html = await CerebrasReportService(get_report_settings()).build_html(
        state["message"]["content"]["content"]
    )
    return {"email_html": html}


async def send_email(state: ReportState) -> dict[str, str]:
    await ResendReportService(get_report_settings()).send_html(
        state.get("email"), state["email_html"]
    )
    return {"email_status": "SENT", "gmail_status": "SENT"}


def finalize_response(state: ReportState) -> dict[str, dict[str, Any]]:
    if state.get("email_status") != "SENT":
        raise RuntimeError("Email delivery did not complete")
    return {
        "response": {
            "job_id": state.get("job_id"),
            "message": "SENT",
            "output": {"base_report": state["email_html"]},
            "report": {"message": state["message"]},
            "gmail_status": "SENT",
        }
    }


def create_report_graph():
    graph = StateGraph(ReportState)
    graph.add_node("normalize_request", normalize_request)
    graph.add_node("build_sections_text", build_sections_text)
    graph.add_node("fetch_finnhub_news", fetch_finnhub_news)
    graph.add_node("fetch_market_context", fetch_market_context)
    graph.add_node("generate_base_report", generate_base_report)
    graph.add_node("enrich_report", enrich_report)
    graph.add_node("normalize_report_envelope", normalize_report_envelope)
    graph.add_node("build_html_email", build_html_email)
    graph.add_node("send_email", send_email)
    graph.add_node("finalize_response", finalize_response)

    graph.add_edge(START, "normalize_request")
    graph.add_edge("normalize_request", "build_sections_text")
    graph.add_edge("build_sections_text", "fetch_finnhub_news")
    graph.add_edge("fetch_finnhub_news", "fetch_market_context")
    graph.add_edge("fetch_market_context", "generate_base_report")
    graph.add_edge("generate_base_report", "enrich_report")
    graph.add_edge("enrich_report", "normalize_report_envelope")
    graph.add_edge("normalize_report_envelope", "build_html_email")
    graph.add_edge("build_html_email", "send_email")
    graph.add_edge("send_email", "finalize_response")
    graph.add_edge("finalize_response", END)
    return graph.compile()


report_graph = create_report_graph()
