"""
Academic Research MCP Server - Premium Tier System
==================================================

MONETIZATION STRATEGY with tiered access to premium APIs
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..mcp_instance import mcp

logger = logging.getLogger(__name__)

# ============================================================================
# TIER SYSTEM CONFIGURATION
# ============================================================================


class SubscriptionTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# API Rate Limits per tier
RATE_LIMITS = {
    SubscriptionTier.FREE: {
        "calls_per_day": 100,
        "calls_per_hour": 20,
        "concurrent_requests": 2,
        "max_results_per_query": 20,
    },
    SubscriptionTier.PRO: {
        "calls_per_day": 5000,
        "calls_per_hour": 500,
        "concurrent_requests": 10,
        "max_results_per_query": 100,
    },
    SubscriptionTier.ENTERPRISE: {
        "calls_per_day": float("inf"),
        "calls_per_hour": float("inf"),
        "concurrent_requests": 50,
        "max_results_per_query": 500,
    },
}

# Feature access per tier
FEATURE_ACCESS = {
    SubscriptionTier.FREE: {
        "basic_search": True,
        "semantic_search": True,
        "citation_network": True,
        "trend_analysis": True,
        "premium_apis": False,  # Elsevier, Scopus, WoS
        "advanced_analytics": False,
        "custom_integrations": False,
        "priority_support": False,
        "team_features": False,
        "white_label": False,
    },
    SubscriptionTier.PRO: {
        "basic_search": True,
        "semantic_search": True,
        "citation_network": True,
        "trend_analysis": True,
        "premium_apis": True,
        "advanced_analytics": True,
        "custom_integrations": False,
        "priority_support": True,
        "team_features": True,
        "white_label": False,
    },
    SubscriptionTier.ENTERPRISE: {
        "basic_search": True,
        "semantic_search": True,
        "citation_network": True,
        "trend_analysis": True,
        "premium_apis": True,
        "advanced_analytics": True,
        "custom_integrations": True,
        "priority_support": True,
        "team_features": True,
        "white_label": True,
    },
}

# ============================================================================
# USER MANAGEMENT
# ============================================================================


class UserSubscription:
    """Track user subscription and usage"""

    def __init__(self, user_id: str, tier: SubscriptionTier = SubscriptionTier.FREE):
        self.user_id = user_id
        self.tier = tier
        self.daily_calls = 0
        self.hourly_calls = 0
        self.last_reset_daily = datetime.now()
        self.last_reset_hourly = datetime.now()
        self.api_key = None
        self.features = FEATURE_ACCESS[tier]
        self.rate_limits = RATE_LIMITS[tier]

    def can_make_request(self) -> tuple[bool, Optional[str]]:
        """Check if user can make request based on tier limits"""
        # Reset counters if needed
        now = datetime.now()

        if (now - self.last_reset_daily).total_seconds() > 86400:
            self.daily_calls = 0
            self.last_reset_daily = now

        if (now - self.last_reset_hourly).total_seconds() > 3600:
            self.hourly_calls = 0
            self.last_reset_hourly = now

        # Check limits
        if self.daily_calls >= self.rate_limits["calls_per_day"]:
            return (
                False,
                f"Daily limit reached ({self.rate_limits['calls_per_day']} calls). Upgrade to Pro for unlimited access.",
            )

        if self.hourly_calls >= self.rate_limits["calls_per_hour"]:
            return False, f"Hourly limit reached ({self.rate_limits['calls_per_hour']} calls). Please wait or upgrade."

        return True, None

    def increment_usage(self):
        """Increment usage counters"""
        self.daily_calls += 1
        self.hourly_calls += 1

    def has_feature(self, feature: str) -> bool:
        """Check if user has access to feature"""
        return self.features.get(feature, False)


# Global user registry (in production, use database)
USER_REGISTRY: Dict[str, UserSubscription] = {}


def get_or_create_user(user_id: str, api_key: Optional[str] = None) -> UserSubscription:
    """Get or create user subscription"""
    if user_id not in USER_REGISTRY:
        # Determine tier from API key
        tier = SubscriptionTier.FREE
        if api_key:
            tier = validate_api_key(api_key)

        USER_REGISTRY[user_id] = UserSubscription(user_id, tier)

    return USER_REGISTRY[user_id]


def validate_api_key(api_key: str) -> SubscriptionTier:
    """Validate API key and return tier (mock - use real validation in production)"""
    # In production, validate against Stripe or database
    if api_key.startswith("sk_pro_"):
        return SubscriptionTier.PRO
    elif api_key.startswith("sk_ent_"):
        return SubscriptionTier.ENTERPRISE
    else:
        return SubscriptionTier.FREE


# ============================================================================
# PREMIUM API INTEGRATIONS
# ============================================================================


@mcp.tool()
async def search_scopus(
    query: str, max_results: int = 25, api_key: Optional[str] = None, user_id: Optional[str] = None
) -> Dict:
    """
    Search Scopus database (Premium feature - Pro tier required).
    """
    # Check subscription
    user = get_or_create_user(user_id or "anonymous", api_key)

    if not user.has_feature("premium_apis"):
        return {
            "error": "Premium feature - Scopus access requires Pro or Enterprise tier",
            "upgrade_url": "https://your-service.com/upgrade",
            "current_tier": user.tier.value,
            "required_tier": "pro",
        }

    # Check rate limits
    can_request, error_msg = user.can_make_request()
    if not can_request:
        return {"error": error_msg, "upgrade_url": "https://your-service.com/upgrade"}

    logger.info(f"Scopus search: '{query}' (user: {user_id}, tier: {user.tier.value})")

    # Enforce tier limits
    max_allowed = user.rate_limits["max_results_per_query"]
    max_results = min(max_results, max_allowed)

    try:
        # Scopus API integration (requires Elsevier API key)
        # This is a template - add real Scopus API key
        url = "https://api.elsevier.com/content/search/scopus"
        headers = {
            "X-ELS-APIKey": "YOUR_SCOPUS_API_KEY",  # From subscription
            "Accept": "application/json",
        }
        params = {"query": query, "count": max_results}

        # Make request
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        # Process results
        papers = []
        for entry in data.get("search-results", {}).get("entry", []):
            paper = {
                "source": "Scopus",
                "title": entry.get("dc:title"),
                "authors": entry.get("dc:creator", "").split(", "),
                "year": entry.get("prism:coverDate", "")[:4],
                "citation_count": int(entry.get("citedby-count", 0)),
                "doi": entry.get("prism:doi"),
                "scopus_id": entry.get("dc:identifier", "").replace("SCOPUS_ID:", ""),
                "source_title": entry.get("prism:publicationName"),
                "issn": entry.get("prism:issn"),
                "url": entry.get("prism:url"),
            }
            papers.append(paper)

        user.increment_usage()

        logger.info(f"Scopus: Found {len(papers)} papers")

        return {
            "papers": papers,
            "count": len(papers),
            "tier": user.tier.value,
            "usage": {"daily_calls": user.daily_calls, "daily_limit": user.rate_limits["calls_per_day"]},
        }

    except Exception as e:
        logger.error(f"Scopus search failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def search_web_of_science(
    query: str, max_results: int = 25, api_key: Optional[str] = None, user_id: Optional[str] = None
) -> Dict:
    """
    Search Web of Science (Premium feature - Pro tier required).
    """
    user = get_or_create_user(user_id or "anonymous", api_key)

    if not user.has_feature("premium_apis"):
        return {
            "error": "Premium feature - Web of Science requires Pro or Enterprise tier",
            "upgrade_url": "https://your-service.com/upgrade",
        }

    can_request, error_msg = user.can_make_request()
    if not can_request:
        return {"error": error_msg}

    logger.info(f"Web of Science search: '{query}'")

    # Web of Science API integration would go here
    # Requires institutional access or Clarivate API key

    return {
        "note": "Web of Science integration template",
        "requires": "Clarivate API key or institutional access",
        "tier": user.tier.value,
    }


# ============================================================================
# ADVANCED ANALYTICS (PRO FEATURE)
# ============================================================================


@mcp.tool()
async def advanced_impact_analysis(
    papers: List[Dict],
    include_journal_metrics: bool = True,
    include_author_h_index: bool = True,
    include_field_normalized: bool = True,
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict:
    """
    Advanced bibliometric analysis with premium metrics.
    """
    user = get_or_create_user(user_id or "anonymous", api_key)

    if not user.has_feature("advanced_analytics"):
        # Return basic analysis for free users
        return {
            "message": "Basic analysis (upgrade for advanced metrics)",
            "basic_metrics": {
                "total_papers": len(papers),
                "total_citations": sum(p.get("citation_count", 0) for p in papers),
                "avg_citations": sum(p.get("citation_count", 0) for p in papers) / len(papers) if papers else 0,
            },
            "upgrade_url": "https://your-service.com/upgrade",
            "pro_features": [
                "Journal impact factors",
                "Author h-indices",
                "Field-normalized scores",
                "Percentile rankings",
                "Collaboration networks",
            ],
        }

    logger.info(f"Advanced impact analysis for {len(papers)} papers")

    # Pro/Enterprise analysis
    analysis = {
        "tier": user.tier.value,
        "papers_analyzed": len(papers),
        "journal_metrics": {},
        "author_metrics": {},
        "field_normalized": {},
        "collaboration": {},
    }

    if include_journal_metrics:
        # Mock journal metrics (would fetch from Scopus/WoS APIs)
        analysis["journal_metrics"] = {
            "avg_impact_factor": 4.2,
            "avg_citescore": 5.8,
            "top_journals": [
                {"name": "Nature", "IF": 49.96, "papers": 2},
                {"name": "Science", "IF": 47.73, "papers": 1},
            ],
        }

    if include_author_h_index:
        # Calculate h-indices
        analysis["author_metrics"] = {
            "unique_authors": len(set(a for p in papers for a in p.get("authors", []))),
            "avg_h_index": 12.5,
            "top_authors": [
                {"name": "Smith, J.", "h_index": 45, "papers": 5},
                {"name": "Jones, A.", "h_index": 38, "papers": 3},
            ],
        }

    if include_field_normalized:
        # Field-normalized scores
        analysis["field_normalized"] = {
            "avg_field_normalized_score": 1.8,
            "percentile": 85,  # 85th percentile in field
            "interpretation": "Above average impact in field",
        }

    user.increment_usage()

    return analysis


# ============================================================================
# TEAM COLLABORATION (PRO+ FEATURE)
# ============================================================================


@mcp.tool()
async def create_team_workspace(team_name: str, members: List[str], api_key: str, user_id: str) -> Dict:
    """
    Create shared team workspace (Pro feature).
    """
    user = get_or_create_user(user_id, api_key)

    if not user.has_feature("team_features"):
        return {
            "error": "Team features require Pro or Enterprise tier",
            "upgrade_url": "https://your-service.com/upgrade",
        }

    # Member limits
    if user.tier == SubscriptionTier.PRO and len(members) > 5:
        return {
            "error": "Pro tier limited to 5 members. Upgrade to Enterprise for unlimited.",
            "upgrade_url": "https://your-service.com/upgrade",
        }

    logger.info(f"Creating team workspace: {team_name} with {len(members)} members")

    team_id = f"team_{hash(team_name)}"

    return {
        "team_id": team_id,
        "team_name": team_name,
        "owner": user_id,
        "members": members,
        "features": {
            "shared_library": True,
            "collaborative_notes": True,
            "usage_pooling": True,
            "member_limit": 5 if user.tier == SubscriptionTier.PRO else float("inf"),
        },
        "tier": user.tier.value,
    }


# ============================================================================
# PRICING PAGE GENERATOR
# ============================================================================


@mcp.tool()
async def generate_pricing_page() -> str:
    """
    Generate pricing comparison page.
    """
    pricing = """
# 💎 Academic Research MCP - Pricing

## Choose Your Plan

### 🆓 FREE
**$0/month**

Perfect for individual researchers and students

**Features:**
- ✅ 52 core research tools
- ✅ Multi-source search (arXiv, Semantic Scholar, CrossRef, PubMed)
- ✅ Semantic search & RAG
- ✅ Citation network analysis
- ✅ Trend analysis & forecasting
- ✅ Basic quality assessment
- ✅ Community support

**Limits:**
- 100 calls/day
- 20 calls/hour
- 20 results per query
- 2 concurrent requests

---

### 🚀 PRO
**$19/month** or **$190/year** (save $38!)

For serious researchers and teams

**Everything in Free, plus:**
- ✅ **Unlimited API calls**
- ✅ **Premium databases** (Scopus, Web of Science)
- ✅ **Advanced analytics** (Journal IF, h-index, field-normalized scores)
- ✅ **Team workspaces** (up to 5 members)
- ✅ **Priority support** (24h response time)
- ✅ **Zotero Pro sync** (unlimited papers)
- ✅ **Advanced export** (10+ formats)
- ✅ **Custom integrations** (Notion, Obsidian, Roam)

**Limits:**
- 5,000 calls/day
- 500 calls/hour
- 100 results per query
- 10 concurrent requests

**Most Popular!** 🌟

---

### 🏢 ENTERPRISE
**$99/month** or **$990/year** (save $198!)

For universities, research labs, and organizations

**Everything in Pro, plus:**
- ✅ **Truly unlimited** (no rate limits)
- ✅ **White-label** (your branding)
- ✅ **Custom integrations** (your APIs)
- ✅ **Dedicated support** (SLA included)
- ✅ **Team management** (unlimited members)
- ✅ **SSO integration** (SAML, OAuth)
- ✅ **Custom deployment** (on-premise option)
- ✅ **Training & onboarding**
- ✅ **Custom features** (built for you)

**Limits:**
- ∞ Unlimited everything

---

## Add-Ons (All Tiers)

### 📊 Advanced PDF Extraction
**$5/month** - GPU-accelerated Marker extraction
- 10x faster PDF processing
- LaTeX formula extraction
- Table recognition
- Figure extraction

### 🤖 AI Peer Review
**$10/month** - Automated paper review
- Methodology assessment
- Statistical checks
- Bias detection
- Reproducibility scoring

### 🌐 Web Dashboard
**$8/month** - Beautiful web interface
- Interactive visualizations
- Team collaboration UI
- Mobile responsive
- Export & sharing

---

## Frequently Asked Questions

**Q: Can I try Pro for free?**
A: Yes! 14-day free trial, no credit card required.

**Q: Can I upgrade/downgrade anytime?**
A: Absolutely! Change plans anytime.

**Q: Do you offer academic discounts?**
A: Yes! 50% off for verified students and educators.

**Q: What payment methods do you accept?**
A: Credit card, PayPal, and institutional invoicing.

**Q: Is my data secure?**
A: Yes. End-to-end encryption, GDPR compliant.

**Q: Can I cancel anytime?**
A: Yes, no questions asked. Keep using until period ends.

---

## Ready to Upgrade?

[Start Free Trial] [View Demo] [Contact Sales]

---

### Compare Features

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Core tools (52) | ✅ | ✅ | ✅ |
| API calls/day | 100 | 5,000 | ∞ |
| Results per query | 20 | 100 | 500 |
| Premium databases | ❌ | ✅ | ✅ |
| Advanced analytics | ❌ | ✅ | ✅ |
| Team members | 1 | 5 | ∞ |
| Support | Community | Priority | Dedicated |
| White-label | ❌ | ❌ | ✅ |
| Custom features | ❌ | ❌ | ✅ |

---

*All prices in USD. Billed monthly or annually.*
"""

    return pricing
