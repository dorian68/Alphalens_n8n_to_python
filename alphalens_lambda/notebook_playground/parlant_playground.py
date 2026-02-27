# %%
import asyncio
import os
import parlant.sdk as p
from typing import List, Sequence
from datetime import datetime, timezone
from dotenv import load_dotenv

# =========================
# TOOLS (placeholders / stubs)
# =========================

load_dotenv()  # Load environment variables from .env file

@p.tool
async def launch_trade_generator(
    context: p.ToolContext,
    instrument: str,
    timeframe: str,
    direction: str = "Both",
    riskLevel: str = "medium",
    strategy: str = "breakout",
    positionSize: str = "2",
    customNotes: str = "",
) -> p.ToolResult:
    """
    Trigger AlphaLens Trade Generator job.
    Replace this stub with your actual backend call (Supabase function / API).
    """
    payload = {
        "tool": "launch_trade_generator",
        "instrument": instrument,
        "timeframe": timeframe,
        "direction": direction,
        "riskLevel": riskLevel,
        "strategy": strategy,
        "positionSize": positionSize,
        "customNotes": customNotes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # TODO: call your real backend here
    return p.ToolResult(data=payload)


@p.tool
async def launch_macro_lab(
    context: p.ToolContext,
    instrument: str,
    timeframe: str = "D1",
    focus: str = "",
    customNotes: str = "",
) -> p.ToolResult:
    """
    Trigger AlphaLens Macro Lab job.
    """
    payload = {
        "tool": "launch_macro_lab",
        "instrument": instrument,
        "timeframe": timeframe,
        "focus": focus,
        "customNotes": customNotes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # TODO: call your real backend here
    return p.ToolResult(data=payload)


@p.tool
async def launch_report(
    context: p.ToolContext,
    instrument: str = "",
    report_type: str = "daily",
    instruments: List[str] = None,
) -> p.ToolResult:
    """
    Trigger AlphaLens report generation.
    """
    payload = {
        "tool": "launch_report",
        "instrument": instrument,
        "report_type": report_type,
        "instruments": instruments or [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # TODO: call your real backend here
    return p.ToolResult(data=payload)


@p.tool
async def get_realtime_price(
    context: p.ToolContext,
    instrument: str,
    dataType: str,  # "quote" | "time_series"
    interval: str = "1day",
    start_date: str = "",
    end_date: str = "",
) -> p.ToolResult:
    """
    Fetch quote or time series (Twelve Data / your API).
    """
    payload = {
        "tool": "get_realtime_price",
        "instrument": instrument,
        "dataType": dataType,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # TODO: call your real market data API here
    return p.ToolResult(data=payload)


@p.tool
async def get_technical_indicators(
    context: p.ToolContext,
    instrument: str,
    indicators: List[str] = None,
    time_period: int = 14,
    interval: str = "1day",
    outputsize: int = 30,
    start_date: str = "",
    end_date: str = "",
) -> p.ToolResult:
    """
    Fetch indicators: RSI, ATR, SMA, EMA, MACD, BBands...
    """
    payload = {
        "tool": "get_technical_indicators",
        "instrument": instrument,
        "indicators": indicators or ["rsi", "sma", "atr", "macd"],
        "time_period": time_period,
        "interval": interval,
        "outputsize": outputsize,
        "start_date": start_date,
        "end_date": end_date,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # TODO: call your real indicators API here
    return p.ToolResult(data=payload)


@p.tool
async def plot_price_chart(
    context: p.ToolContext,
    instrument: str,
    interval: str,
    lookback_hours: float,
) -> p.ToolResult:
    """
    Ask frontend/backend to render a chart for a given instrument/window.
    """
    payload = {
        "tool": "plot_price_chart",
        "instrument": instrument,
        "interval": interval,
        "lookback_hours": lookback_hours,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # TODO: produce chart payload for frontend
    return p.ToolResult(data=payload)


@p.tool
async def get_collective_intelligence(
    context: p.ToolContext,
    instrument: str = "",
    lookback_days: int = 30,
) -> p.ToolResult:
    """
    Collective intelligence (community setups + macro + ABCG) anonymized.
    This maps your Supabase collective-insights logic.
    """
    payload = {
        "tool": "get_collective_intelligence",
        "instrument": instrument,
        "lookback_days": lookback_days,
        "privacy": "anonymized_only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    # TODO: call Supabase functions and aggregate results
    return p.ToolResult(data=payload)


# =========================
# OPTIONAL PREAMBLE CONTROL
# =========================

async def preamble_instruction_provider(ctx: p.EngineContext) -> Sequence[str]:
    hour = datetime.now().hour
    if hour < 12:
        return ["Use a concise morning-style acknowledgment."]
    if hour >= 18:
        return ["Use a concise evening-style acknowledgment."]
    return ["Use a short professional acknowledgment before tool work."]


# =========================
# AGENT BOOTSTRAP
# =========================

async def build_aura_agent(server: p.Server) -> p.Agent:
    aura = await server.create_agent(
        id="aura_alphalens",
        name="AURA",
        description=(
            "AlphaLens Unified Research Assistant. A specialized financial markets AI assistant "
            "for FX, crypto, macro analysis, technical analysis, and trade preparation. "
            "Professional, concise, and action-oriented. Prioritizes risk management and factual responses. "
            "Uses collective intelligence in anonymized form and never exposes personal user data."
        ),
        preamble_config=p.PreambleConfiguration(
            examples=["One moment, I’m checking that.", "Let me analyze this.", "Checking now."],
            get_instructions=preamble_instruction_provider,
        ),
        # output_mode=p.OutputMode.STREAM,  # optional
    )

    # -------------------------
    # CORE BEHAVIOR GUIDELINES
    # -------------------------

    # Language protocol (default English, strict switching)
    await aura.create_guideline(
        condition="The user writes in English or the language is ambiguous",
        action="Respond in English by default",
        description=(
            "English is the default language. Only switch language when the user clearly writes "
            "in another language with explicit phrasing or directly requests another language."
        ),
        criticality=p.Criticality.HIGH,
        track=False,
    )

    await aura.create_guideline(
        condition="The user clearly writes in French, Spanish, or German",
        action="Respond entirely in the user's language for the current reply",
        description=(
            "Maintain consistency for the whole response. If the user switches back to English, "
            "switch back to English immediately."
        ),
        criticality=p.Criticality.HIGH,
        track=False,
    )

    # Anti-hallucination
    await aura.create_guideline(
        condition="The user asks for market analysis, numbers, setups, or trends",
        action="Only use available tool data and provided context, and never invent numbers or facts",
        description=(
            "If exact data is unavailable, explicitly say the data is unavailable. Avoid vague numerical "
            "claims like 'around' or 'approximately' when no verified value exists."
        ),
        criticality=p.Criticality.HIGH,
        track=False,
    )

    await aura.create_guideline(
        condition="The response references market data or collective insights",
        action="State the basis of the claim and cite how many datapoints or which source was used",
        description=(
            "Examples: 'Based on X community setups analyzed...' / 'According to ABCG insights...' / "
            "'Based on the latest technical indicator values...'"
        ),
        criticality=p.Criticality.HIGH,
        track=False,
    )

    # Privacy / collective intelligence
    await aura.create_guideline(
        condition="The user asks about community sentiment, collective trends, or recent setups",
        action="Use only anonymized aggregated community trends and never expose user identifiers",
        description=(
            "Use phrasing like 'community analysis shows...' and 'recent setups indicate...'. "
            "Never reveal user_id or personal information."
        ),
        criticality=p.Criticality.HIGH,
        tools=[get_collective_intelligence],
        track=False,
    )

    # Style / communication quality
    await aura.create_guideline(
        condition="You are answering a financial or trading question",
        action="Be concise, actionable, and emphasize risk management",
        description=(
            "Use professional financial terminology, avoid unnecessary verbosity, and always frame "
            "trade-related responses with risk-awareness."
        ),
        criticality=p.Criticality.MEDIUM,
        track=False,
    )

    # -------------------------
    # INTENT ROUTING GUIDELINES
    # -------------------------

    # Trade setup routing
    await aura.create_guideline(
        condition="The user asks for a trade setup, trade idea, trade signal, or entry stop-loss take-profit levels",
        action="Collect the required trade parameters and then launch the trade generator",
        description=(
            "Required minimum: instrument and timeframe. Default reasonable optional values if omitted "
            "(riskLevel=medium, strategy=breakout, positionSize=2). Do not ask for confirmation once the "
            "required information is present."
        ),
        criticality=p.Criticality.HIGH,
        tools=[launch_trade_generator],
    )

    # Macro routing
    await aura.create_guideline(
        condition="The user asks for macro commentary, market outlook, macro analysis, or what is happening with a market",
        action="Collect the required instrument and then launch macro analysis",
        description=(
            "Use macro lab for macro outlook and commentary. Do not use macro lab for entry levels or trade setup requests."
        ),
        criticality=p.Criticality.HIGH,
        tools=[launch_macro_lab],
    )

    # Report routing
    await aura.create_guideline(
        condition="The user asks for a market report, portfolio report, daily report, or weekly report",
        action="Collect required instruments and report type and then launch report generation",
        description="Do not ask for confirmation after required parameters are available.",
        criticality=p.Criticality.HIGH,
        tools=[launch_report],
    )

    # Chart plotting routing
    await aura.create_guideline(
        condition="The user asks to plot, chart, graph, draw, or visualize a price over a time window",
        action="Parse the instrument and lookback window and generate a price chart",
        description=(
            "If no explicit lookback is given, default to 24 hours. Pick an interval appropriate to the window."
        ),
        criticality=p.Criticality.MEDIUM,
        tools=[plot_price_chart],
    )

    # Technical analysis routing (dual-tool pattern)
    await aura.create_guideline(
        condition="The user asks for technical analysis, indicators, RSI, moving averages, ATR, MACD, or technical setup",
        action="Fetch price time series and technical indicators, then synthesize a technical analysis report",
        description=(
            "Default indicators: RSI, SMA, ATR, MACD. Mention analyzed timeframe and current UTC context. "
            "Present trend, momentum, volatility, MACD signal, key levels, and trading bias."
        ),
        criticality=p.Criticality.HIGH,
        tools=[get_realtime_price, get_technical_indicators],
    )

    # -------------------------
    # SPECIAL POLICY GUIDELINES
    # -------------------------

    # Political / geopolitical reframing for macro-trade requests
    await aura.create_guideline(
        condition="The user asks for macro or trading analysis involving politicians or geopolitical scenarios",
        action="Reframe the request objectively as policy scenarios and continue the analysis workflow",
        description=(
            "Never refuse purely because politicians are mentioned. Rephrase to neutral policy scenario language "
            "while preserving the user's instrument and analytical intent."
        ),
        criticality=p.Criticality.HIGH,
        track=False,
    )

    # Missing data behavior
    await aura.create_guideline(
        condition="A required market datapoint or indicator is unavailable",
        action="Acknowledge the missing data gracefully and continue with available evidence",
        description=(
            "Do not expose raw technical errors. Explain what is unavailable and still provide value from the remaining data."
        ),
        criticality=p.Criticality.MEDIUM,
        track=False,
    )

    print(f"AURA agent created: {aura.id}")
    return aura

async def main():
    host = os.getenv("PARLANT_HOST", "0.0.0.0")
    port = int(os.getenv("PARLANT_PORT", "8800"))

    async with p.Server(
        nlp_service=p.NLPServices.openai,
        host=host,
        port=port,
    ) as server:
        await build_aura_agent(server)

        public_host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
        print(f"Parlant UI: http://{public_host}:{port}/chat")

if __name__ == "__main__":
    asyncio.run(main())
# %%
