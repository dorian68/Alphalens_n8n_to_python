# %%
from typing import TypedDict, Optional, Dict, Any
import json

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# =========================
# 1️⃣ STATE DEFINITION
# =========================

class AgentState(TypedDict):
    messages: list
    tool_request: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]

# =========================
# 2️⃣ TOOL (pure Python)
# =========================

@tool
async def forecast_aws_api(symbol: str, timeframe: str, horizons: list, trade_mode: Optional[str] = "forward") -> dict:
    """
    Call the AWS forecast API to generate trade ideas and forecasts
    for a given symbol, timeframe, and list of horizons.
    """
    
    start = time.time()
    import httpx

    FORECAST_URL = "https://jqrlegdulnnrpiixiecf.supabase.co/functions/v1/forecast-proxy"
    print(f"[TOOL_FORECAST] Fetching market price for {symbol}")
    payload = {
        "body": {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizons": horizons,
            "trade_mode": trade_mode,
            "use_montecarlo": True,
            "include_predictions": True,
            "include_metadata": True,
            "include_model_info": True,
            "paths": 1000
            }
    }
    

    async with httpx.AsyncClient() as client:
        r = await client.post(FORECAST_URL, json=payload, timeout=30)
        r.raise_for_status()
        end = time.time()
        print(f"Forecast AWS API call completed in {end - start:.2f} seconds.")
        # print("Forecast AWS API response:", r.json())
        return r.json()

def get_market_price(symbol: str) -> dict:
    print(f"[TOOL] Fetching market price for {symbol}")
    return {
        "symbol": symbol,
        "price": 2345.67,
        "currency": "USD"
    }

# =========================
# 3️⃣ LLM AGENT – DECIDE TOOL OR FINAL
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def agent_reasoning(state: AgentState) -> AgentState:
    prompt = """
You are a trading assistant helping building trading ideas by sharing prices forecast and TP SLsuggestions.

If you need a market price to answer, respond ONLY with JSON:
{"tool":"get_market_price","args":{"symbol":"XAU/USD"}}

If you already know enough, respond with:
FINAL: <your answer>

if you need to forecast prices level and trade ideas with Take Profit and Stop Loss levels, respond ONLY with JSON:
{"tool":"forecast_aws_api","args":{"symbol":"AAPL","timeframe":"1d","horizons":[7,14,30]}}
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    content = response.content.strip()
    print("[LLM reasoning]:", content)

    if content.startswith("{"):
        state["tool_request"] = json.loads(content)
    elif content.startswith("FINAL"):
        state["final_answer"] = content.replace("FINAL:", "").strip()

    return state

# =========================
# 4️⃣ TOOL EXECUTOR NODE
# =========================

def tool_executor(state: AgentState) -> AgentState:
    req = state["tool_request"]

    if not req:
        return state

    if req["tool"] == "get_market_price":
        result = get_market_price(**req["args"])
        state["tool_result"] = result
    elif req["tool"] == "forecast_aws_api":
        import asyncio
        result = asyncio.run(forecast_aws_api(**req["args"]))
        state["tool_result"] = result
    return state

# =========================
# 5️⃣ LLM AFTER TOOL
# =========================

def agent_after_tool(state: AgentState) -> AgentState:
    result = state["tool_result"]

    response = llm.invoke([
        HumanMessage(
            content=f"""
result from tool call:
{result}

Could you let me know at what level will be traded EUR/USD in 7, 14 and 30 hours ? 
"""
        )
    ])

    state["final_answer"] = response.content.strip()
    return state

# =========================
# 6️⃣ ROUTER (THE KEY PART)
# =========================

def router(state: AgentState):
    if state.get("tool_request"):
        return "tool_executor"
    return END

# =========================
# 7️⃣ BUILD GRAPH
# =========================

graph = StateGraph(AgentState)

graph.add_node("agent_reasoning", agent_reasoning)
graph.add_node("tool_executor", tool_executor)
graph.add_node("agent_after_tool", agent_after_tool)

graph.set_entry_point("agent_reasoning")

graph.add_conditional_edges(
    "agent_reasoning",
    router,
    {
        "tool_executor": "tool_executor",
        END: END,
    }
)

graph.add_edge("tool_executor", "agent_after_tool")
graph.add_edge("agent_after_tool", END)

agent = graph.compile()

# =========================
# 8️⃣ RUN
# =========================

#%%
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [],
        "tool_request": None,
        "tool_result": None,
        "final_answer": None
    })

    print("\n================ FINAL ANSWER ================")
    print(result["final_answer"])

# %%
