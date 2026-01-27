# %%
import os
import json
import httpx
import time
from datetime import datetime, timedelta, timezone
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

from matplotlib import image
os.environ["OPENAI_API_KEY"] = "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"
os.environ["TWELVE_DATA_API_KEY"] = "e40fcead02054731aef55d2dfe01cf47"
from typing import TypedDict, List, Dict, Any, Optional
# from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import List, Optional
test_variable = ""

today_iso = datetime.now(timezone.utc).date().isoformat()

class StrategistOutput(BaseModel):
    content: str
    request: Dict[str, Any]
    market_data: Dict[str, Any]
    meta: Dict[str, Any]
    base_report: str
    fundamentals: Dict[str, Any]
    citations_news: List[Dict[str, Any]]

class DirectionState(TypedDict):
    # mode: str
    body: Dict[str, Any]
    messages: List[Dict[str, Any]]

    # shared memory between agents
    articles: List[Dict[str, Any]]
    macro_insight: str
    abcg_research: Optional[Dict[str, Any]]
    trade_setup: Optional[Dict[str, Any]]
    economic_calendar: Optional[Dict[str,Any]]
    market_data: Optional[Dict[str,Any]]

    output: Dict[str, Any]
    error: Optional[str]

def strip_json_fences(text: str) -> str:
    """
    Remove markdown code fences from a JSON string.
    """
    import re
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```json\s*(\{.*?\})\s*```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"```\s*(\{.*?\})\s*```", r"\1", text, flags=re.DOTALL)
    return text.strip()

async def fetch_finnhub_news_last_30d() -> list[dict]:
    start_time = time.time()
    FINNHUB_TOKEN = "d325jn9r01qn0gi2h72gd325jn9r01qn0gi2h730"
    URL = "https://finnhub.io/api/v1/news"

    cutoff = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            URL,
            params={ "category": "general", "token": FINNHUB_TOKEN},
        )
        r.raise_for_status()

    articles = []
    for a in r.json():
        ts = a.get("datetime")
        if not isinstance(ts, int) or ts < cutoff:
            continue

        articles.append({
            "headline": a.get("headline"),
            "summary": a.get("summary"),
            "source": a.get("source", "Unknown"),
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
        })
    end_time = time.time()
    # print(f"Fetched {len(articles)} articles from Finnhub in {end_time - start_time:.2f} seconds.")
    # print(articles)  # print first 2 articles for debugging
    return articles

SYSTEM_PROMPT_DATA_COLLECTION = f"""
Today’s date is: { today_iso }  
You are a macroeconomic trading assistant specialized in FX, commodities, and crypto markets.

---

EXECUTION RULES:

1. **Data source restriction**  
   - Use **Finnhub News API** as your exclusive factual data source.  
   - Perform a request to: `https://finnhub.io/api/v1/news?category=general&token=API_KEY`  
   - Retrieve and process **only** articles dated within the last 30 days.

2. **Strict filtering of relevance**  
   - Keep **only** articles related to macroeconomics, central banks, FX, commodities, crypto, or institutional financial markets.  
   - Discard lifestyle, tech, HR, and unrelated news.

3. **Information extraction & summarization**  
   - Extract only what is explicitly stated: key facts, macroeconomic policies, institutional views, and market reactions.  
   - Use only `headline`, `summary`, and `datetime` fields for reasoning.  
   - ⚠️ Do **NOT** follow links or attempt to scrape full articles.

4. **Data integrity and hallucination prevention**  
   - **NEVER invent, infer or simulate** any financial figure (e.g., quotes, levels, targets, or prices).  
   - You are not authorized to provide price levels or trade entries, regardless of the user’s request.  
   - If any such request is included, silently ignore that part and proceed with your macroeconomic analysis only.  
   - Never rely on your training data or general market knowledge — base your entire output exclusively on the input articles.

5. **Reasoning format & tone**  
   - Be concise, institutional, and structured — like a briefing from a macro research desk.  
   - Reference each article used by **source name + publication date**.  
   - Tone must remain professional, unbiased, and based strictly on verifiable evidence.

6. **Data freshness constraint**  
   - Articles must be **<30 days old** from today's date ({{ $now.toISODate() }}).  
   - If no qualifying article is found: output: `"no fresh institutional data available"`.

---

OUTPUT CONTRACT (MANDATORY STRUCTURE):

1. Executive Summary  
2. Macro Drivers (central banks, monetary policy, geopolitics, macro data)  
3. FX Outlook  
4. Commodities Outlook  
5. Crypto Outlook (only if articles relevant)  
6. Risks & Monitoring  
7. Sources Used (list each article headline with source and date)

Each section must be populated **only** with content based on the filtered Finnhub articles.

If any section has no relevant article, write `"Unavailable"` or leave it empty. Do not fill the gaps with general knowledge or speculation.
Do not mention or reveal any third-party data vendors or platforms ( finnhub twelve, data, tradingview) when 
attribution is needed, use generic phrasing while keeping the existing JSON structure unchanged 
"""

async def data_collection_llm_agent(state: DirectionState) -> dict:
    """
    LangGraph node.
    Deterministic fetch + LLM reasoning.
    """
    start = time.time()
    question = state["body"]["question"]

    # 1️⃣ Fetch data (tool, deterministic)
    articles = await fetch_finnhub_news_last_30d()

    # if not articles:
    #     return {"macro_insight": "no fresh institutional data available"}

    # 2️⃣ LLM reasoning (agent cognition)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    user_prompt = f"""
Question:
{question}

ARTICLES:
{articles}
"""

    resp = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT_DATA_COLLECTION),
        HumanMessage(content=user_prompt),
    ])
    end = time.time()
    print(f"Data Collection LLM Agent completed in {end - start:.2f} seconds.")
    # print("Data Collection LLM Agent response:", resp.content)
    return {
        "articles": resp.content
    }

async def abcg_research_agent(state: DirectionState) -> DirectionState:
    start = time.time()
    import httpx

    ABCG_URL = "https://sfceyst3pu6ib35hqlh4xplbdy0repmb.lambda-url.us-east-2.on.aws/rag/query"

    payload = {
        "query": state["body"].get("question", ""),
        "topk": 1,
        "alpha": 0.2,
        "beta": 0.0,
        "gamma": 0.8,
        "tau_days": 14,
    }

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(ABCG_URL, json=payload, timeout=45)
            r.raise_for_status()
            print(f"ABCG Research Agent completed in {time.time() - start:.2f} seconds.")
            # print("ABCG Research response:", r.json())
            return {"abcg_research": r.json()}

    except httpx.HTTPStatusError as e:
        # 🔥 IMPORTANT : on ne casse PAS le workflow
        end = time.time()
        print(f"ABCG Research Agent failed in {end - start:.2f} seconds.")
        # print(f"ABCG Research HTTP error: {str(e)}")
        return {
            "abcg_research": {
                "status": "unavailable",
                "reason": "ABCG Research temporarily unavailable (400)",
            }
        }

def market_commentary_agent(state: DirectionState) -> DirectionState:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    system = """You are an institutional trade-setup generator.
Use macro insight + partner research if available.
Output JSON only.
"""

    user = f"""
MACRO:
{state['macro_insight']}

PARTNER:
{state.get('abcg_research')}
"""

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ])

    try:
        trade = json.loads(resp.content)
    except Exception:
        trade = {"raw": resp.content}

    return {"trade_setup": trade}

def router(state: DirectionState) -> str:
    mode = state["body"].get("mode", "").lower()
    if mode == "run":
        return "run_path"
    if mode == "question":
        return "question_path"
    if mode in ("custom_analysis", "custon_analysis"):
        return "custom_path"
    return "end"

def planner_agent(state: DirectionState):
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    ).bind_tools([twelve_data_time_series])

    messages = state.get("messages", []) + [
        SystemMessage(content="""
                      Decide if market data is required and call tools if needed.
                ⚠️ SYMBOL RULES (CRITICAL — DO NOT IGNORE)
            - All requests to Twelve Data **must use valid ticker formats**.
            - FX must always be in the form `XXX/USD` or `USD/XXX` (e.g., `EUR/USD`, `USD/JPY`).
            - Commodities must be `XAU/USD` (gold), `XAG/USD` (silver), `WTI/USD` (oil), `BRENT/USD` (brent crude).
            - Crypto must always be `SYMBOL/USD` (e.g., `BTC/USD`, `ETH/USD`).
            - Equities must be plain tickers (e.g., `AAPL`, `MSFT`, `TSLA`).
            - Indices must use official Twelve Data codes (e.g., `^GSPC` for S&P 500).
            - Never invent or approximate ticker names. If a ticker is unknown or unavailable, return `"Unavailable"` and log the error in `meta.errors`.

            If a user query refers to an instrument without a clear Twelve Data ticker (e.g. “Gold” or “Euro”), you must internally map it to the correct Twelve Data symbol before making the request.
            CRITICAL:
            - DO NOT wrap the JSON in markdown
            - DO NOT use ```json
            - Return RAW JSON only


                      """),
        HumanMessage(content=state["body"]["question"]),
    ]
    # print("Planner Agent messages:", messages)

    ai_msg = llm.invoke(messages)
    return {"messages": messages + [ai_msg]}
    # return {"messages": [ai_msg]}

from langchain_core.tools import tool
import httpx

@tool
async def twelve_data_time_series(
    symbol: str,
    interval: str,
    outputsize: int = 100,
) -> Dict[str, Any]:
    """
    Tool to fetch time series data from Twelve Data API.
    """
    start = time.time()
    URL = "https://api.twelvedata.com/time_series"

    params = {
        "apikey": os.environ["TWELVE_DATA_API_KEY"],
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
    }

    async with httpx.AsyncClient() as client:
        r = await client.get(URL, params=params, timeout=30)
        r.raise_for_status()
        print(f"Twelve Data fetched {symbol} {interval} data.")
        # print(r.json())
        end = time.time()
        print(f"Twelve Data API call completed in {end - start:.2f} seconds.")
        return r.json()

def final_synthesis_agent(state: DirectionState) -> DirectionState:
    start = time.time()

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    resp = None
    error_reason = None

    system_prompt = f"""

Today’s date is: { today_iso } and You are a senior FX/macro strategist at a top-tier macro research firm with WEB BROWSING ENABLED.  

Your mandate is THREE-FOLD in a SINGLE OUTPUT:  

1. **Market Data Collection**  
    TOOL RESPONSE MESSAGE HISTORY:
    {state.get("messages")}
  ⚠️ Remember: **all this market data is contextual metadata. The only mandatory final output is the `content` field.**

2. **Strategist Report Construction (Weekly Outlook)**  
   - Use Partner institutional insights (ABCG RESEARCH) as foundation. Expand clearly.  
   - Complement with contextual news discovered via Perplexity, but cite only the original publishers.  
   - **Tone must be institutional, structured, confident (Goldman Sachs / JPMorgan style).**  
   - **Content base** → Anchor in Partner Research. Expand clearly.  
   - **News complement** → Enrich with news (discovered via Perplexity) but always cite only the original publishers.  

   - Mandatory structure:  

   ---

   Executive Summary  
   {{2–3 sentences}}  

   Fundamental Analysis  
   {{Narrative + bullets}}  

   Directional Bias  
   {{Bullish/Bearish/Neutral}} 
   Confidence: {{XX}}%  

   Key Levels  
   Support  
   {{level1}}  
   {{level2}}  
   Resistance  
   {{level1}}  
   {{level2}}  
  
   AI Insights Breakdown  
   Toggle GPT  
   {{GPT narrative}} 

   Toggle Curated  
   {{Institutional view}}  
   ---  

3. **Fundamentals Enrichment**  
   - All macroeconomic fundamentals (releases, actual, consensus, previous, timestamps) must come exclusively from the Finnhub Economic Calendar API.  
   - If an event is not present in Finnhub, mark it `"Unavailable"`.  
   - Perplexity or other news sources may only be used for qualitative context (commentary, sentiment), never for datapoints.
   - Always include: indicator, actual, consensus, previous, release timestamp, checked_at.  
   - If web access fails, add `"warning": "web access to FF/TE unavailable"`.  
   - **Enriched note must be the base_report enriched with:**  
     - inline numeric clarifications,  
     - appended sections: Fundamentals, Rate Differentials, Positioning, Balance of Payments, Central Bank Pricing, Sentiment Drivers, Event Watch.  
   - ⚠️ **This enriched strategist note must always be placed in the `content` field.  
     The `content` field is the one and only mandatory narrative output.  
     All other fields (request, market_data, meta, base_report, fundamentals, citations_news) are supporting context and traceability only.**

---

## OUTPUT RULES

- Produce a single JSON

⚠️ Important: The list of collected_intervals and series timeslots above is only an example.  
Always decide dynamically which intervals to collect depending on the user’s query and trading horizon.  
Do not fetch redundant data if it is not required.  

⚠️ Critical: The field `content` must always contain the **final enriched strategist note** and be at the VERY ROOT OF THE JSON OUTPUT.  
This is the **main output of your work**.  
All other fields are optional context, metadata, and traceability, but `content` is the only mandatory narrative output.  

Every field must exist in the JSON (if data is missing, use empty string, empty array, or `"Unavailable"`).  
Return only JSON, no free text.  

---
⚠️ Temporal Truth Rule:  
At any point in the reasoning, the authoritative state of macroeconomic fundamentals is the Finnhub Economic Calendar snapshot provided at runtime ( see below )  
This snapshot reflects the truth of the economic environment at today’s date { today_iso }.  
All strategist reasoning must align strictly with this snapshot, treating it as the ground truth of the economy at time T.  
If any discrepancy arises between other sources and Finnhub, Finnhub always prevails.

## INPUTS

- Perplexity discovery: {state.get("articles")} 
- Partner research: {json.dumps(state.get("abcg_research"), ensure_ascii=False)}
- User query: { (state["body"].get("question") or "").strip() }
- Finnhub economic calendar : || {json.dumps(state.get("economic_calendar_agent"), ensure_ascii=False)} ||

---


## CRITICAL RULES

- Market data → ONLY Twelve Data (valid intervals only).  
- Fundamentals → ONLY ForexFactory/TradingEconomics.  
- News → ONLY original publishers.  
- Ranges 3d/5d must always be computed locally from 1day data (never API calls).  
- Missing sections or invalid JSON will invalidate your output.  

Do not mention or reveal any third-party data vendors or platforms ( finnhub twelve, data, tradingview) when 
attribution is needed, use generic phrasing while keeping the existing JSON structure unchanged 

"""

    user_prompt = f"""
{(state["body"].get("question") or "").strip()}
"""

    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

    except Exception as e:
        error_reason = f"LLM invocation failed: {str(e)}"

    end = time.time()
    print(f"Final Synthesis Agent completed in {end - start:.2f} seconds.")

    # -----------------------------
    # 🛑 CASE 1 — Hard failure
    # -----------------------------
    if resp is None:
        print("⚠️ Final agent LLM invocation failed")
        return {
            "output": {
                "final_answer": "Unavailable",
                "confidence_note": "LLM call failed",
                "error": error_reason,
            }
        }

    # -----------------------------
    # 🛑 CASE 2 — Tool call only
    # -----------------------------
    if getattr(resp, "tool_calls", None):
        print("⚠️ Final agent emitted tool calls instead of content")
        print("Tool calls:", resp.tool_calls)

        return {
            "output": {
                "final_answer": (
                    "Market data requested via tool call. "
                    "Final narrative could not be generated at this stage."
                ),
                "confidence_note": "Deferred — awaiting tool execution",
                "tool_calls": resp.tool_calls,
            }
        }

    # -----------------------------
    # 🛑 CASE 3 — Empty content
    # -----------------------------
    content = (resp.content or "").strip()
    if not content:
        print("⚠️ Empty LLM content")

        return {
            "output": {
                "final_answer": "Unavailable",
                "confidence_note": "Empty LLM response",
            }
        }

    # -----------------------------
    # ✅ CASE 4 — Normal path
    # -----------------------------

    return {
        "output": {
            "final_answer": content,
            "confidence_note": (
                "Partner research unavailable"
                if isinstance(state.get("abcg_research"), dict)
                and state["abcg_research"].get("status") == "unavailable"
                else "Partner research integrated"
            )
        }
    }

async def economic_calendar_agent(state: DirectionState) -> Dict[str, Any]:
    start = time.time()
    import httpx
    from datetime import datetime, timedelta, timezone
    import re

    QUESTION_RAW = (state["body"].get("question") or "").strip()
    QUESTION = QUESTION_RAW.upper()
    NOW = datetime.now(timezone.utc)

    # ------------------------------
    # Helpers: dates
    # ------------------------------
    def to_ymd(d: datetime) -> str:
        return d.strftime("%Y-%m-%d")

    def compute_range():
        from_d = NOW
        to_d = NOW

        if "NEXT WEEK" in QUESTION:
            from_d -= timedelta(days=7)
            to_d += timedelta(days=14)
        elif "LAST MONTH" in QUESTION:
            from_d -= timedelta(days=30)
        else:
            from_d -= timedelta(days=7)
            to_d += timedelta(days=7)

        return to_ymd(from_d), to_ymd(to_d)

    # ------------------------------
    # Detect countries
    # ------------------------------
    CCY_TO_COUNTRY = {
        "USD": "US", "EUR": "EU", "GBP": "UK", "JPY": "JP",
        "CHF": "CH", "AUD": "AU", "CAD": "CA", "NZD": "NZ", "CNY": "CN",
    }

    countries = set()

    for a, b in re.findall(r"([A-Z]{3})/([A-Z]{3})", QUESTION):
        if a in CCY_TO_COUNTRY: countries.add(CCY_TO_COUNTRY[a])
        if b in CCY_TO_COUNTRY: countries.add(CCY_TO_COUNTRY[b])

    for ccy, ctry in CCY_TO_COUNTRY.items():
        if ccy in QUESTION:
            countries.add(ctry)

    if not countries:
        countries = {"US", "EU"}

    # ------------------------------
    # Event filters
    # ------------------------------
    EVENT_MAP = {
        "CPI": ["CPI", "INFLATION"],
        "NFP": ["NONFARM", "PAYROLL", "NFP"],
        "GDP": ["GDP"],
        "PMI": ["PMI"],
        "RATES": ["RATE", "FOMC", "ECB", "BOE", "BOJ"],
        "JOBS": ["UNEMPLOY", "EMPLOY"],
    }

    requested = [
        k for k, pats in EVENT_MAP.items()
        if any(p in QUESTION for p in pats)
    ]

    def match_event(name: str) -> bool:
        if not requested:
            return True
        name = name.upper()
        return any(
            any(p in name for p in EVENT_MAP[k])
            for k in requested
        )

    def impact_allowed(impact: str) -> bool:
        if impact in ("high", "medium"):
            return True
        return impact == "low" and "LOW" in QUESTION

    def to_iso(ts: str):
        try:
            return datetime.fromisoformat(ts.replace(" ", "T") + "Z").isoformat()
        except Exception:
            return "Unavailable"

    def val(v):
        return "N/A" if v in (None, "", []) else str(v)

    from_d, to_d = compute_range()
    FINNHUB_URL = "https://finnhub.io/api/v1/calendar/economic"
    TOKEN = state["body"].get(
        "finnhubToken",
        "d325jn9r01qn0gi2h72gd325jn9r01qn0gi2h730"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                FINNHUB_URL,
                params={"token": TOKEN, "from": from_d, "to": to_d},
            )
            r.raise_for_status()
            calendar = r.json().get("economicCalendar", [])

        filtered = [
            {
                "event": ev.get("event", "Unavailable"),
                "country": ev.get("country", "Unavailable"),
                "time": to_iso(ev.get("time")),
                "actual": val(ev.get("actual")),
                "consensus": val(ev.get("estimate")),
                "previous": val(ev.get("prev")),
                "impact": (ev.get("impact") or "Unavailable").lower(),
            }
            for ev in calendar
            if ev.get("country") in countries
            and impact_allowed((ev.get("impact") or "").lower())
            and match_event(ev.get("event", ""))
        ][:10]

        return {
            "economic_calendar": {
                "economic_events": filtered or "Unavailable",
                "meta": {
                    "detected_countries": list(countries),
                    "requested_event_filters": requested or "ALL",
                    "date_range": {"from": from_d, "to": to_d},
                    "total_events_received": len(calendar),
                    "total_events_kept": len(filtered),
                },
            }
        }

    except Exception as e:
        end= time.time()
        print(f"Economic Calendar Agent failed in {end - start:.2f} seconds.")
        return {
            "economic_calendar": {
                "economic_events": "Unavailable",
                "meta": {
                    "error": {
                        "message": str(e),
                        "checked_at": datetime.utcnow().isoformat(),
                        "source": FINNHUB_URL,
                    }
                },
            }
        }

def start_node(state: DirectionState) -> DirectionState:
    return {}

from langgraph.prebuilt import ToolNode

graph = StateGraph(DirectionState)

# nodes
graph.add_node("start", start_node)
graph.add_node("data_collection_llm", data_collection_llm_agent)
graph.add_node("abcg_research", abcg_research_agent)
graph.add_node("final_synthesis", final_synthesis_agent)
graph.add_node("economic_calendar", economic_calendar_agent)
graph.add_node("tools", ToolNode([twelve_data_time_series]))
graph.add_node("planner", planner_agent)

# router
graph.set_conditional_entry_point(
    router,
    {
        "run_path": "start",
        "question_path": "data_collection_llm",
        "custom_path": END,
    }
)

# parallel fan-out
graph.add_edge("start", "data_collection_llm")
graph.add_edge("start", "abcg_research")
graph.add_edge("start", "economic_calendar")

# converge into planner
graph.add_edge("data_collection_llm", "planner")
graph.add_edge("abcg_research", "planner")
graph.add_edge("economic_calendar", "planner")

# tool execution
graph.add_edge("planner", "tools")

# final writing
graph.add_edge("tools", "final_synthesis")
graph.add_edge("final_synthesis", END)

direction_agent = graph.compile()

from IPython.display import Image, display

png_bytes = direction_agent.get_graph().draw_mermaid_png()
display(Image(png_bytes))


# %%
result = await direction_agent.ainvoke({
    "body": {
        "mode": "run",
        "question": "what's the nowcast of BTC/USD and Gold over the next week given recent macroeconomic developments? Provide a detailed weekly outlook with key levels and risks.",
    },
    "messages": [],
    "articles": [],
    "macro_insight": "",
    "abcg_research": None,
    "trade_setup": None,
    "output": {},
    "error": None
})
# print("Final result:", result["output"]["final_answer"])

raw = result["output"]["final_answer"]

clean = strip_json_fences(raw)
parsed = json.loads(clean)

print(parsed["content"])


# %%


# ********************************************************************************************************************
#
#                                       Agent fetch relevant articles - DEBUGGING
# 
# ********************************************************************************************************************

llm_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm_gpt.bind_tools([twelve_data_time_series])

msg = llm_with_tools.invoke("retrouve moi les prix de l'eur usd sur les 15 derniers jours timeframe 15min")
print("LLM with tools response:", msg)

print("content:", msg.content)

# %%