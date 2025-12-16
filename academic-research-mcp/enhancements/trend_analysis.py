"""
Academic Research MCP Server - Trend Analysis & Forecasting Module
==================================================================

Features:
- Analyze research trends over time
- Compare multiple topics/keywords
- Detect emerging topics
- Forecast future research directions
- Generate trend visualizations
- Identify "hot" vs "cooling" research areas
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
except ImportError:
    print("Installing trend analysis dependencies...")
    import subprocess

    subprocess.check_call(["pip", "install", "pandas", "matplotlib", "seaborn", "scipy", "numpy"])
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy import stats

from ..mcp_instance import mcp
from ..research_tools import search_semantic_scholar

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ============================================================================
# TOOL 1: ANALYZE RESEARCH TRENDS
# ============================================================================


@mcp.tool()
async def analyze_research_trends(
    topic: str, years_back: int = 10, max_papers_per_year: int = 50, sources: Optional[List[str]] = None
) -> Dict:
    """
    Analyze how research on a topic has evolved over time.
    """
    logger.info(f"Analyzing trends for '{topic}' over {years_back} years")

    current_year = datetime.now().year
    start_year = current_year - years_back

    # Collect papers by year
    yearly_data = {}
    all_keywords = []

    for year in range(start_year, current_year + 1):
        logger.info(f"Fetching papers for {year}...")

        try:
            papers = await search_semantic_scholar(
                query=topic, max_results=max_papers_per_year, year_filter=f"{year}-{year}"
            )

            yearly_data[year] = {
                "count": len(papers),
                "total_citations": sum(p.get("citation_count", 0) for p in papers),
                "avg_citations": sum(p.get("citation_count", 0) for p in papers) / len(papers) if papers else 0,
                "papers": papers,
            }

            # Extract keywords from titles
            for paper in papers:
                title = paper.get("title", "").lower()
                words = [w for w in title.split() if len(w) > 4]
                all_keywords.extend(words)

        except Exception as e:
            logger.error(f"Error fetching {year}: {e}")
            yearly_data[year] = {"count": 0, "total_citations": 0, "avg_citations": 0, "papers": []}

    # Calculate growth metrics
    years = list(yearly_data.keys())
    counts = [yearly_data[y]["count"] for y in years]

    if len(years) > 2 and sum(counts) > 0:
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)

        # Calculate percentage growth
        avg_count = sum(counts) / len(counts)
        annual_growth_rate = (slope / avg_count * 100) if avg_count > 0 else 0

        # Trend direction
        if slope > 0 and p_value < 0.05:
            trend = "increasing"
        elif slope < 0 and p_value < 0.05:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        slope = r_value = annual_growth_rate = 0
        trend = "insufficient_data"

    # Analyze keyword evolution
    midpoint = start_year + (years_back // 2)
    recent_keywords = []
    older_keywords = []

    for year, data in yearly_data.items():
        for paper in data["papers"]:
            words = [w for w in paper.get("title", "").lower().split() if len(w) > 4]
            if year >= midpoint:
                recent_keywords.extend(words)
            else:
                older_keywords.extend(words)

    # Find emerging topics
    recent_freq = Counter(recent_keywords)
    older_freq = Counter(older_keywords)

    emerging_topics = []
    for keyword, recent_count in recent_freq.most_common(50):
        older_count = older_freq.get(keyword, 0)
        if recent_count > older_count * 2 and recent_count >= 5:  # 2x growth minimum
            growth_factor = recent_count / max(older_count, 1)
            emerging_topics.append(
                {
                    "keyword": keyword,
                    "recent_mentions": recent_count,
                    "older_mentions": older_count,
                    "growth_factor": round(growth_factor, 2),
                }
            )

    # Sort by growth factor
    emerging_topics.sort(key=lambda x: x["growth_factor"], reverse=True)

    # Find peak year
    peak_year = max(yearly_data.items(), key=lambda x: x[1]["count"])[0]

    result = {
        "topic": topic,
        "time_period": f"{start_year}-{current_year}",
        "total_papers": sum(data["count"] for data in yearly_data.values()),
        "yearly_breakdown": yearly_data,
        "growth_metrics": {
            "annual_growth_rate_percent": round(annual_growth_rate, 2),
            "r_squared": round(r_value**2, 3),
            "trend": trend,
            "slope": round(slope, 2),
        },
        "peak_year": peak_year,
        "peak_count": yearly_data[peak_year]["count"],
        "emerging_topics": emerging_topics[:10],
        "top_keywords": [{"keyword": k, "frequency": f} for k, f in Counter(all_keywords).most_common(20)],
    }

    logger.info(
        f"Trend analysis complete: {result['total_papers']} papers, "
        f"{annual_growth_rate:.1f}% annual growth, trend: {trend}"
    )

    return result


# ============================================================================
# TOOL 2: COMPARE RESEARCH TOPICS
# ============================================================================


@mcp.tool()
async def compare_research_topics(topics: List[str], years_back: int = 5, max_papers: int = 30) -> Dict:
    """
    Compare trends across multiple research topics.
    """
    logger.info(f"Comparing {len(topics)} topics over {years_back} years")

    comparison_data = {}

    for topic in topics:
        logger.info(f"Analyzing: {topic}")

        try:
            trend = await analyze_research_trends(topic=topic, years_back=years_back, max_papers_per_year=max_papers)

            # Calculate momentum (recent vs older papers)
            current_year = datetime.now().year
            recent_count = sum(
                trend["yearly_breakdown"].get(y, {}).get("count", 0) for y in range(current_year - 1, current_year + 1)
            )

            comparison_data[topic] = {
                "total_papers": trend["total_papers"],
                "growth_rate": trend["growth_metrics"]["annual_growth_rate_percent"],
                "trend": trend["growth_metrics"]["trend"],
                "peak_year": trend["peak_year"],
                "recent_momentum": recent_count,
                "r_squared": trend["growth_metrics"]["r_squared"],
            }

        except Exception as e:
            logger.error(f"Error analyzing '{topic}': {e}")
            comparison_data[topic] = None

    # Filter valid results
    valid_topics = {k: v for k, v in comparison_data.items() if v is not None}

    if not valid_topics:
        return {"error": "No valid data for any topic", "topics_analyzed": len(topics)}

    # Create rankings
    rankings = {
        "by_total_papers": sorted(valid_topics.items(), key=lambda x: x[1]["total_papers"], reverse=True),
        "by_growth_rate": sorted(valid_topics.items(), key=lambda x: x[1]["growth_rate"], reverse=True),
        "by_recent_momentum": sorted(valid_topics.items(), key=lambda x: x[1]["recent_momentum"], reverse=True),
    }

    # Generate insights
    insights = []

    # Hottest topic
    if rankings["by_growth_rate"]:
        hottest = rankings["by_growth_rate"][0]
        insights.append(f"'{hottest[0]}' shows the highest growth rate at {hottest[1]['growth_rate']:.1f}% annually.")

    # Most established
    if rankings["by_total_papers"]:
        established = rankings["by_total_papers"][0]
        insights.append(
            f"'{established[0]}' is the most established with {established[1]['total_papers']} total papers."
        )

    # Current momentum
    if rankings["by_recent_momentum"]:
        momentum = rankings["by_recent_momentum"][0]
        insights.append(
            f"'{momentum[0]}' has the strongest current momentum with "
            f"{momentum[1]['recent_momentum']} papers in the last 2 years."
        )

    # Identify declining topics
    declining = [topic for topic, data in valid_topics.items() if data["trend"] == "decreasing"]
    if declining:
        insights.append(f"Declining interest in: {', '.join(declining)}")

    result = {
        "topics_compared": len(topics),
        "time_period": f"{datetime.now().year - years_back}-{datetime.now().year}",
        "data": comparison_data,
        "rankings": rankings,
        "insights": insights,
    }

    logger.info(f"Topic comparison complete for {len(topics)} topics")

    return result


# ============================================================================
# TOOL 3: FORECAST RESEARCH DIRECTION
# ============================================================================


@mcp.tool()
async def forecast_research_direction(topic: str, years_forward: int = 3, confidence_level: float = 0.8) -> Dict:
    """
    Forecast future research directions based on current trends.
    """
    logger.info(f"Forecasting research direction for '{topic}'")

    # Get historical trends
    trend = await analyze_research_trends(topic=topic, years_back=10, max_papers_per_year=50)

    # Extract time series
    years = sorted(trend["yearly_breakdown"].keys())
    counts = [trend["yearly_breakdown"][y]["count"] for y in years]

    if len(years) < 3:
        return {"error": "Insufficient historical data for forecasting", "years_available": len(years)}

    # Linear regression for forecast
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)

    # Generate forecast
    current_year = datetime.now().year
    forecast_years = list(range(current_year + 1, current_year + years_forward + 1))
    forecast_counts = [int(max(0, slope * year + intercept)) for year in forecast_years]

    # Calculate confidence
    r_squared = r_value**2

    if r_squared >= 0.8:
        confidence_label = "High"
        reliability = "Strong"
    elif r_squared >= 0.5:
        confidence_label = "Medium"
        reliability = "Moderate"
    else:
        confidence_label = "Low"
        reliability = "Weak"

    # Generate recommendations based on trend
    growth_rate = trend["growth_metrics"]["annual_growth_rate_percent"]

    recommendations = []

    if growth_rate > 20:
        recommendations.append("🔥 Rapidly growing field. Early contributions could be highly impactful.")
    elif growth_rate > 10:
        recommendations.append("📈 Growing field with good opportunities for novel research.")
    elif growth_rate > 0:
        recommendations.append("📊 Steady growth. Look for underexplored niches.")
    else:
        recommendations.append("📉 Declining or saturated field. Consider novel angles or pivot.")

    if r_squared < 0.5:
        recommendations.append("⚠️ Volatile trend. High-risk, high-reward opportunities may exist.")

    # Highlight emerging sub-topics
    if trend["emerging_topics"]:
        top_emerging = trend["emerging_topics"][0]["keyword"]
        recommendations.append(f"💡 '{top_emerging}' is emerging as a key sub-topic. Consider specializing here.")

    result = {
        "topic": topic,
        "forecast_period": f"{current_year + 1}-{current_year + years_forward}",
        "historical_period": trend["time_period"],
        "forecasted_papers": dict(zip(forecast_years, forecast_counts)),
        "confidence": round(r_squared, 3),
        "confidence_label": confidence_label,
        "reliability": reliability,
        "growth_rate_percent": round(growth_rate, 2),
        "trend_direction": trend["growth_metrics"]["trend"],
        "emerging_topics": trend["emerging_topics"][:5],
        "recommendations": recommendations,
    }

    logger.info(f"Forecast complete: {confidence_label} confidence ({r_squared:.2%} R²)")

    return result


# ============================================================================
# TOOL 4: VISUALIZE TREND
# ============================================================================


@mcp.tool()
async def visualize_research_trend(
    trend_data: Dict, output_file: str = "research_trend.png", include_forecast: bool = False
) -> str:
    """
    Create publication-quality trend visualization.
    """
    logger.info("Generating trend visualization...")

    # Extract data
    yearly = trend_data["yearly_breakdown"]
    years = sorted(yearly.keys())
    counts = [yearly[y]["count"] for y in years]
    citations = [yearly[y]["avg_citations"] for y in years]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Paper count over time
    ax1.plot(years, counts, marker="o", linewidth=2, markersize=8, color="#2E86AB", label="Actual")
    ax1.fill_between(years, counts, alpha=0.3, color="#2E86AB")

    # Add trend line
    z = np.polyfit(years, counts, 1)
    p = np.poly1d(z)
    ax1.plot(years, p(years), "--", color="red", alpha=0.8, linewidth=2, label="Trend")

    # Forecast (if requested)
    if include_forecast:
        forecast_years = [max(years) + i for i in range(1, 4)]
        forecast_counts = [int(p(y)) for y in forecast_years]
        ax1.plot(forecast_years, forecast_counts, ":", color="orange", linewidth=2, marker="s", label="Forecast")

    ax1.set_title(f"Research Output Trend: {trend_data['topic']}", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Number of Papers", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Average citations over time
    ax2.bar(years, citations, color="#A23B72", alpha=0.7)
    ax2.set_title("Average Citations per Paper", fontsize=14)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Average Citations", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Visualization saved to {output_file}")
    return f"Trend visualization saved to: {output_file}"


# ============================================================================
# TOOL 5: IDENTIFY HOT VS COOLING TOPICS
# ============================================================================


@mcp.tool()
async def identify_hot_vs_cooling(
    topics: List[str], threshold_hot: float = 15.0, threshold_cooling: float = -5.0
) -> Dict:
    """
    Categorize topics as "hot" (growing), "stable", or "cooling" (declining).
    """
    logger.info(f"Categorizing {len(topics)} topics...")

    results = {"hot": [], "stable": [], "cooling": [], "insufficient_data": []}

    for topic in topics:
        try:
            trend = await analyze_research_trends(topic, years_back=5)
            growth_rate = trend["growth_metrics"]["annual_growth_rate_percent"]

            data = {
                "topic": topic,
                "growth_rate": growth_rate,
                "total_papers": trend["total_papers"],
                "trend": trend["growth_metrics"]["trend"],
            }

            if growth_rate >= threshold_hot:
                results["hot"].append(data)
            elif growth_rate <= threshold_cooling:
                results["cooling"].append(data)
            else:
                results["stable"].append(data)

        except Exception as e:
            logger.warning(f"Could not analyze '{topic}': {e}")
            results["insufficient_data"].append(topic)

    # Sort each category
    results["hot"].sort(key=lambda x: x["growth_rate"], reverse=True)
    results["cooling"].sort(key=lambda x: x["growth_rate"])
    results["stable"].sort(key=lambda x: x["total_papers"], reverse=True)

    logger.info(
        f"Categorization complete: {len(results['hot'])} hot, "
        f"{len(results['stable'])} stable, {len(results['cooling'])} cooling"
    )

    return results
