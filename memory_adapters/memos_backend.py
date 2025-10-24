# memory_adapters/memos_backend.py
from typing import List, Dict, Any
import os
import requests
import threading
import numpy as np

# 轻量语义检索：sentence-transformers + sklearn（cosine）
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MemosBackend:
    """
    基于 memos 的后端适配器：
    - 本地维护一个简单的语义索引（SentenceTransformer + cosine），保证可复现的向量检索；
    - 可选：把内容同步到 memos 服务器（HTTP API），便于审计/留痕；
    - 暴露统一接口：add(records), retrieve(query, k), clear()
    """
    def __init__(self, namespace: str = "ehr", emb_model: str = "all-MiniLM-L6-v2"):
        self.namespace = namespace

        # 本地内存 & 向量索引
        self._items: List[Dict[str, Any]] = []     # [{"text": str, "meta": {...}, "id": int}, ...]
        self._embeddings: np.ndarray = None        # shape: (N, dim)
        self._lock = threading.Lock()

        # 嵌入模型
        self._model = SentenceTransformer(emb_model)

        # 可选：同步到 memos 服务器
        self._memos_base = os.getenv("MEMOS_BASE_URL", "").rstrip("/")
        self._memos_token = os.getenv("MEMOS_API_KEY", "")
        self._sync_enabled = bool(self._memos_base and self._memos_token)

    # ------------------- public APIs -------------------
    def add(self, records: List[Dict[str, Any]]):
        """
        records: [{"text": "...", "meta": {...}}, ...]
        """
        texts = []
        with self._lock:
            start_id = len(self._items)
            for i, r in enumerate(records):
                txt = (r.get("text") or "").strip()
                meta = r.get("meta", {}) or {}
                if not txt:
                    continue
                rid = start_id + i
                self._items.append({"id": rid, "text": txt, "meta": meta})
                texts.append(txt)

            # 更新向量索引
            if texts:
                vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                if self._embeddings is None:
                    self._embeddings = vecs
                else:
                    self._embeddings = np.vstack([self._embeddings, vecs])

        # 可选：同步写入 memos（不影响检索）
        if self._sync_enabled and texts:
            self._sync_to_memos(texts)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = query.strip()
        if not q or self._embeddings is None or len(self._items) == 0:
            return []
        qvec = self._model.encode([q], convert_to_numpy=True, normalize_embeddings=True)  # (1, dim)
        sims = cosine_similarity(qvec, self._embeddings)[0]                               # (N,)
        # 取 top-k
        idxs = np.argsort(-sims)[:k]
        out = []
        for idx in idxs:
            item = self._items[idx]
            out.append({"text": item["text"], "meta": {"score": float(sims[idx]), **item.get("meta", {})}})
        return out

    def clear(self):
        with self._lock:
            self._items.clear()
            self._embeddings = None

    # ------------------- optional sync -------------------
    def _sync_to_memos(self, texts: List[str]):
        """
        把文本作为 memo 同步到 memos（可选，不影响检索）
        参考 REST：POST {base}/api/v1/memo  (不同部署版本端点可能略有差异)
        """
        url = f"{self._memos_base}/api/v1/memo"
        headers = {"Authorization": self._memos_token, "Content-Type": "application/json"}
        for t in texts:
            payload = {"content": t}
            try:
                requests.post(url, headers=headers, json=payload, timeout=5)
            except Exception:
                # 同步失败不影响主流程
                pass
