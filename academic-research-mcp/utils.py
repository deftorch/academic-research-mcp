import hashlib
import logging
import os
from typing import Any, Dict, Optional

import httpx
from diskcache import Cache
from fuzzywuzzy import fuzz
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mcp_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ===== API ENDPOINTS =====
ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
CROSSREF_API = "https://api.crossref.org/works"
CORE_API = "https://api.core.ac.uk/v3"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ===== CONFIGURATION =====
DEFAULT_EMAIL = "your.email@university.edu"
CONTACT_EMAIL = os.environ.get("ACADEMIC_CONTACT_EMAIL", DEFAULT_EMAIL)
CACHE_TTL = 3600  # 1 hour
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# ===== PERSISTENT CACHE =====
CACHE_DIR = os.path.join(os.getcwd(), ".cache_data")
_cache = Cache(CACHE_DIR)
# Configure default expiration time for cache items
_cache.expire = CACHE_TTL

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class ProviderError(Exception):
    """Base exception for Search Provider errors"""

    pass


class NetworkError(Exception):
    """Raised when network issues occur (connection, timeout)"""

    pass


class PaperNotFoundError(ProviderError):
    """Raised when no papers found from any source"""

    pass


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails"""

    pass


class RateLimitExceeded(NetworkError):
    """Raised when API rate limit is hit"""

    pass


class APIError(ProviderError):
    """Base exception for API errors"""

    pass


class ConfigurationError(Exception):
    """Raised when server configuration is invalid"""

    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_config():
    """Validate server configuration on startup"""
    if CONTACT_EMAIL == DEFAULT_EMAIL:
        logger.warning(
            "⚠️  SECURITY WARNING: Using default CONTACT_EMAIL. "
            "Please set ACADEMIC_CONTACT_EMAIL env var to avoid API blocking."
        )


def cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate cache key from function name and arguments"""
    key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_from_cache(key: str) -> Optional[Any]:
    """Get value from cache if exists and not expired"""
    # DiskCache handles expiry automatically
    value = _cache.get(key)
    if value is not None:
        logger.debug(f"Cache hit: {key}")
        return value
    return None


def set_to_cache(key: str, value: Any, expire: int = CACHE_TTL):
    """Set value to cache"""
    _cache.set(key, value, expire=expire)
    logger.debug(f"Cache set: {key}")


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate fuzzy similarity between two texts (0-1)"""
    if not text1 or not text2:
        return 0.0
    return fuzz.ratio(text1.lower().strip(), text2.lower().strip()) / 100.0


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    return " ".join(text.split()).strip()


# ============================================================================
# API CLIENT WITH RETRY LOGIC
# ============================================================================


import asyncio

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
)
async def make_api_request(
    url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: int = REQUEST_TIMEOUT
) -> Dict:
    """Make HTTP request with automatic retry and exponential backoff"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, params=params, headers=headers)

            if response.status_code == 429:
                logger.warning(f"Rate limit hit for {url}")
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                        logger.info(f"Waiting {wait_time}s due to Retry-After header")
                        await asyncio.sleep(wait_time)
                        # Recursive retry after waiting
                        return await make_api_request(url, params, headers, timeout)
                    except ValueError:
                        pass # Retry-After might be a date, or parsing failed

                raise RateLimitExceeded(f"Rate limit exceeded for {url}")

            if response.status_code == 404:
                logger.warning(f"Resource not found: {url}")
                return {}

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            else:
                return {"text": response.text}

        except httpx.HTTPStatusError as e:
            # Handle 429 specifically if it wasn't caught above (though it should be)
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit for {url} (caught in HTTPStatusError)")
                raise RateLimitExceeded(f"Rate limit exceeded for {url}")

            logger.error(f"HTTP error {e.response.status_code}: {url}")
            raise APIError(f"HTTP {e.response.status_code}: {url}")
        except httpx.TimeoutException:
            logger.error(f"Request timeout: {url}")
            raise NetworkError(f"Request timed out: {url}")
        except httpx.RequestError as e:
            logger.error(f"Network error: {str(e)} - {url}")
            raise NetworkError(f"Network connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"Request failed: {str(e)} - {url}")
            raise APIError(f"Unknown error: {str(e)}")
