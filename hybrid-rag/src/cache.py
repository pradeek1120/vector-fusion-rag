import hashlib
import json
from typing import Optional

import redis

from config.settings import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_redis_client: Optional[redis.Redis] = None


def get_redis() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            _redis_client.ping()
            logger.info("redis_connected")
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e))
            _redis_client = None
    return _redis_client


def _cache_key(query: str, top_k: Optional[int], use_hyde: Optional[bool]) -> str:
    raw = f"{query.strip().lower()}|{top_k}|{use_hyde}"
    return "rag:v1:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached(
    query: str,
    top_k: Optional[int] = None,
    use_hyde: Optional[bool] = None,
) -> Optional[dict]:
    r = get_redis()
    if not r:
        return None
    try:
        key = _cache_key(query, top_k, use_hyde)
        data = r.get(key)
        if data:
            logger.info("cache_hit", key=key[:20])
            return json.loads(data)
    except Exception as e:
        logger.warning("cache_get_error", error=str(e))
    return None


def set_cached(
    query: str,
    result: dict,
    top_k: Optional[int] = None,
    use_hyde: Optional[bool] = None,
):
    r = get_redis()
    if not r:
        return
    try:
        key = _cache_key(query, top_k, use_hyde)
        # Don't cache error responses
        if result.get("error"):
            return
        r.setex(key, settings.cache_ttl_seconds, json.dumps(result))
        logger.info("cache_set", key=key[:20], ttl=settings.cache_ttl_seconds)
    except Exception as e:
        logger.warning("cache_set_error", error=str(e))
