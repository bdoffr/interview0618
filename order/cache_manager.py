import logging

import redis
import hashlib
from django.conf import settings

logger = logging.getLogger('django')


class CacheManager:
    def __init__(self):
        self.redis_client = redis.StrictRedis.from_url(settings.REDIS_URL, decode_responses=True)

    def _get_search_key(self, query):
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"search:{query_hash}"

    def get_search_results(self, query):
        key = self._get_search_key(query)
        try:
            # 转换redis存储的字符串
            result_str = self.redis_client.get(key)
            return [int(pid) for pid in result_str.split(',')] if result_str else None
        except:
            logger.error(f"results: {redis.exceptions.RedisError}")
            return None  # 降级处理

    def set_search_results(self, query, product_ids, ttl=300):  # 默认5分钟
        key = self._get_search_key(query)
        # 将列表转换为逗号分隔的字符串
        value = ",".join(map(str, product_ids))
        try:
            self.redis_client.set(key, value, ex=ttl)
        except:
            logger.error(f"results: {redis.exceptions.RedisError}")
            pass  # 降级处理
