# memory_adapters/amem_backend.py

from typing import List, Dict, Any
import os

class AMemBackend:
    """
    A-mem 替换为 AgenticMemorySystem 后端适配器：
    - 首选用 OpenAI 做 LLM 分析（如你示例所示）
    - 若没 OPENAI_API_KEY 或初始化失败，则降级为“无 LLM 分析”模式：用简单规则生成关键词，仍可检索
    """
    def __init__(self, namespace: str = "ehr", emb_model: str = "all-MiniLM-L6-v2"):
        self.namespace = namespace
        self._llm_mode = "none"
        self._init_memory_system(emb_model)

        # 用一个简单的本地索引备份（降级模式下也能 work）
        self._fallback_store: List[Dict[str, Any]] = []

    def _init_memory_system(self, emb_model: str):
        try:
            # 包名可能是 agentic_memory 或 agentic_memory；上层已经 pip 安装两种可能之一
            from agentic_memory.memory_system import AgenticMemorySystem  # type: ignore

            # 优先按你的示例：openai + gpt-4o-mini
            llm_backend = "openai" if os.getenv("OPENAI_API_KEY") else None
            llm_model = "gpt-4o-mini" if llm_backend == "openai" else None

            self.ms = AgenticMemorySystem(
                model_name=emb_model,     # 用于向量检索（ChromaDB）
                llm_backend=llm_backend,  # "openai" 或 None
                llm_model=llm_model       # "gpt-4o-mini" 或 None
            )
            self._llm_mode = "openai" if llm_backend == "openai" else "none"
        except Exception as e:
            # 完全降级到“无 LLM”的兜底实现
            self.ms = None
            self._llm_mode = "none"

    # ---- public API ----
    def add(self, records: List[Dict[str, Any]]):
        """
        records: [{"text": "...", "meta": {...}}, ...]
        """
        for r in records:
            content = r.get("text", "")
            meta = r.get("meta", {}) or {}

            if self.ms is not None:
                # 三种添加方式：自动、半自动、手动。这里按有/无 LLM 决定。
                if self._llm_mode == "openai":
                    # 全自动（让 LLM 生成 keywords/context/tags）
                    self.ms.add_note(content)
                else:
                    # 无 LLM：手工生成一些简易关键词，仍可建立可用向量索引
                    kw = self._heuristic_keywords(content)
                    self.ms.add_note(content, keywords=kw, context=meta.get("context"), tags=meta.get("tags"))
            else:
                # 完全降级到内存列表（仍保证 retrieve 接口可用）
                self._fallback_store.append({"text": content, "meta": meta})

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.ms is not None:
            try:
                # 主检索：embedding 语义检索
                hits = self.ms.search(query, k=k)
                out = []
                for h in hits:
                    out.append({
                        "text": h.get("content", h.get("text", "")),
                        "meta": {
                            "keywords": h.get("keywords"),
                            "tags": h.get("tags"),
                            "score": h.get("score")
                        }
                    })
                return out
            except Exception:
                pass

        # 兜底：朴素检索（关键词匹配 + 简单打分）
        if not self._fallback_store:
            return []
        q_terms = set(self._heuristic_keywords(query))
        scored = []
        for item in self._fallback_store:
            terms = set(self._heuristic_keywords(item["text"]))
            score = len(q_terms & terms)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k]]

    def clear(self):
        if self.ms is not None:
            # AgenticMemorySystem 一般提供 delete/reset；若没有就忽略
            try:
                # 如果作者暴露了 delete / reset api，可在此调用
                pass
            except Exception:
                pass
        self._fallback_store.clear()

    # ---- helpers ----
    @staticmethod
    def _heuristic_keywords(text: str, top_n: int = 8) -> List[str]:
        import re
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        stop = set([
            "the","a","an","and","or","to","of","in","on","for","with","at","by","is","are",
            "this","that","it","as","be","from","was","were","will","can","could","should","would"
        ])
        freq = {}
        for t in tokens:
            if t.isdigit() or t in stop or len(t) <= 2:
                continue
            freq[t] = freq.get(t, 0) + 1
        return [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
