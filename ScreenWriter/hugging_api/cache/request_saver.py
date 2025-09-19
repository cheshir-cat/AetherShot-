# request_saver.py
from __future__ import annotations
import os, json, sqlite3, hashlib, time
from typing import Any, Callable, Dict, Optional, Tuple, List, Protocol

# ──────────────────────────────────────────────────────────────────────────────
# Ключ по содержимому запроса (model + messages + extras)
def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def make_cache_key(model: str, messages: List[Dict[str, str]], extras: Optional[Dict[str, Any]] = None) -> str:
    payload = {"model": model, "messages": messages, "extras": extras or {}}
    return hashlib.sha256(_stable_dumps(payload).encode("utf-8")).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# Дисковый кэш (SQLite)
class DiskCache:
    def __init__(self, path: str = "./.cache/llm_cache.sqlite") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            con.commit()

    def get(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.path) as con:
            cur = con.execute("SELECT response FROM cache WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, response: str) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?, ?, ?)",
                        (key, response, time.time()))
            con.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Семантический кэш (ChromaDB + локальные эмбеддинги)
try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None  # позволяем жить без хромы (только дисковый кэш)

class EmbeddingBackend(Protocol):
    def embed(self, text: str) -> List[float]: ...

class LocalEmbeddingBackend:
    """Локальные эмбеддинги без API-стоимости."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

class SemanticCache:
    def __init__(
        self,
        persist_dir: str = "./.cache/chroma",
        collection: str = "request_cache",
        embedding_backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        if chromadb is None:
            raise RuntimeError("chromadb is not installed. `pip install chromadb`")
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_backend = embedding_backend or LocalEmbeddingBackend()
        self.coll = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def get(self, prompt: str, threshold: float = 0.86) -> Optional[Dict[str, Any]]:
        qemb = self.embedding_backend.embed(prompt)
        res = self.coll.query(query_embeddings=[qemb], n_results=1, include=["documents", "metadatas", "distances"])
        if not res["ids"] or not res["ids"][0]:
            return None
        dist = res["distances"][0][0]  # cosine distance (0 — идентично)
        sim = 1 - dist
        if sim >= threshold:
            meta = res["metadatas"][0][0] or {}
            payload = meta.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = None
            return {"similarity": sim, "payload": payload}
        return None

    def put(self, prompt: str, payload: Dict[str, Any]) -> None:
        emb = self.embedding_backend.embed(prompt)
        _id = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        self.coll.upsert(
            ids=[_id],
            embeddings=[emb],
            documents=[prompt],
            metadatas=[{"payload": json.dumps(payload, ensure_ascii=False)}],
        )

# ──────────────────────────────────────────────────────────────────────────────
# Декоратор кэширования
def cache_llm_call(
    disk_cache: Optional[DiskCache] = None,
    semantic_cache: Optional[SemanticCache] = None,
    response_to_str: Callable[[Any], str] = lambda x: json.dumps(x, ensure_ascii=False),
    response_from_str: Optional[Callable[[str], Any]] = None,
    key_fn: Callable[[str, List[Dict[str, str]], Optional[Dict[str, Any]]], str] = make_cache_key,
    semantic_prompt_fn: Callable[[List[Dict[str, str]]], str] = lambda m: m[-1]["content"],  # по умолчанию юзер-промпт
):
    """
    Оборачивает функцию вида f(model, messages, **kwargs) -> response_obj.

    - disk_cache: SQLite-кэш (по хэшу запроса).
    - semantic_cache: ChromaDB-кэш (по смысловой близости последнего user-промпта).
    - response_to_str / response_from_str: сериализация и десериализация результата.
    - key_fn: формирование ключа для дискового кэша.
    - semantic_prompt_fn: какой текст считать «запросом» для эмбеддингов (по умолчанию последняя user-реплика).
    """
    disk_cache = disk_cache or DiskCache()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(model: str, messages: List[Dict[str, str]], *args, **kwargs) -> Any:
            extras = kwargs.get("extras")  # произвольные влияющие на ответ параметры
            key = key_fn(model, messages, extras)

            # 1) Дисковый кэш
            cached = disk_cache.get(key)
            if cached is not None:
                return response_from_str(cached) if response_from_str else json.loads(cached)

            # 2) Семантический кэш (необязательный)
            if semantic_cache is not None:
                query_text = semantic_prompt_fn(messages)
                sem_hit = semantic_cache.get(query_text)
                if sem_hit and sem_hit.get("payload") is not None:
                    payload = sem_hit["payload"]
                    # Сохраняем как обычный кэш — чтобы следующий раз брать по точному ключу
                    disk_cache.set(key, response_to_str(payload))
                    return payload if response_from_str is None else response_from_str(response_to_str(payload))

            # 3) Зовём реальный API
            response_obj = func(model, messages, *args, **kwargs)

            # 4) Сохраняем
            s = response_to_str(response_obj)
            disk_cache.set(key, s)

            if semantic_cache is not None:
                try:
                    query_text = semantic_prompt_fn(messages)
                    semantic_cache.put(query_text, json.loads(s))
                except Exception:
                    pass

            return response_obj
        return wrapper
    return decorator
