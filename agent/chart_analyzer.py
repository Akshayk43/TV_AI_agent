"""Vision-based chart analysis using Claude's multimodal capabilities."""

import base64
import anthropic

from config.settings import ANTHROPIC_API_KEY, MODEL_NAME
from agent.prompts.system_prompts import CHART_ANALYSIS_PROMPT


def analyze_chart_image(
    image_b64: str,
    context: str = "",
    media_type: str = "image/png",
) -> str:
    """Send a chart image to Claude for visual analysis.

    Args:
        image_b64: Base64-encoded chart image.
        context: Additional context about the chart (symbol, timeframe, etc.).
        media_type: MIME type of the image.

    Returns:
        Claude's detailed chart analysis as a string.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_b64,
            },
        },
        {
            "type": "text",
            "text": (
                f"Analyze this candlestick chart in detail.\n\n"
                f"Context: {context}\n\n"
                "Provide:\n"
                "1. Current trend direction and strength\n"
                "2. Key support/resistance levels visible\n"
                "3. Candlestick patterns in recent bars\n"
                "4. Chart patterns (triangles, H&S, flags, etc.)\n"
                "5. Indicator readings if visible\n"
                "6. Volume analysis if visible\n"
                "7. Volume profile analysis if visible (POC, VAH, VAL)\n"
                "8. Overall bias (bullish/bearish/neutral) with confidence level\n"
                "9. Suggested trade setup if a high-probability one exists"
            ),
        },
    ]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        system=CHART_ANALYSIS_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    return response.content[0].text


def analyze_chart_with_data(
    image_b64: str,
    data_summary: str,
    indicator_summary: str,
    volume_profile_summary: str,
    candlestick_summary: str,
    price_action_summary: str,
) -> str:
    """Comprehensive chart analysis combining vision with computed data.

    Args:
        image_b64: Base64-encoded chart image.
        data_summary: Raw data summary string.
        indicator_summary: Technical indicator summary.
        volume_profile_summary: Volume profile analysis.
        candlestick_summary: Detected candlestick patterns.
        price_action_summary: Price action analysis.

    Returns:
        Comprehensive analysis combining visual and quantitative data.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    context_text = f"""## Computed Analysis Data

{indicator_summary}

{volume_profile_summary}

{price_action_summary}

{candlestick_summary}

## Instructions
You have both the visual chart AND computed analysis data above. Combine both sources
to provide a comprehensive analysis. The visual chart may reveal patterns that the
computed data misses (like chart patterns, visual confluence zones). The computed data
provides precise values the visual cannot.

Synthesize everything into:
1. **Market Regime**: Trending or ranging? Strong or weak?
2. **Key Levels**: Most important S/R levels with confluence
3. **Pattern Recognition**: Both candlestick and chart patterns
4. **Indicator Confluence**: Where multiple indicators agree
5. **Volume Analysis**: What volume tells us about conviction
6. **Trade Recommendation**: Direction, entry zone, stop-loss, targets
7. **Confidence Score**: 1-10, with explanation
"""

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64,
            },
        },
        {
            "type": "text",
            "text": context_text,
        },
    ]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=6000,
        system=CHART_ANALYSIS_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    return response.content[0].text
