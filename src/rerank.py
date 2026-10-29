"""CrossEncoder reranker.

Lazy-loaded, offline-capable. Enable by setting ``MEMORY_RERANKER_MODEL`` (which
also flips ``rerank_enabled`` in ``config.load_config``). The model weights are
baked into the image at build time (see ``Dockerfile``), so no network call is
made at runtime.

Pipeline: recall ``rerank_recall_n`` candidates from the fused ranking -> score
each (query, doc) pair with the CrossEncoder -> keep ``max(top_k, rerank_top_k)``.
Defaults: off; ``rerank_recall_n=300``, ``rerank_top_k=100``.
"""
from __future__ import annotations

import threading

import numpy as np
from functools import lru_cache


class Reranker:
    """Thin wrapper around ``sentence_transformers.CrossEncoder``."""

    def __init__(self, model_name: str, device: str | None = None,
                 batch_size: int = 32, max_length: int = 512):
        # 延迟导入：无 torch 环境 import 本模块不会直接崩
        from sentence_transformers import CrossEncoder

        if batch_size <= 0:
            raise ValueError("reranker batch_size must be positive")
        if max_length <= 0:
            raise ValueError("reranker max_length must be positive")
        self.model = CrossEncoder(model_name, device=device,
                                  max_length=max_length,
                                  trust_remote_code=False)
        self.batch_size = batch_size
        self._lock = threading.RLock()

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float64)
        pairs = [(query, d) for d in docs]
        with self._lock:
            scores = self.model.predict(pairs, batch_size=self.batch_size,
                                        show_progress_bar=False,
                                        convert_to_numpy=True)
        return np.asarray(scores, dtype=np.float64).reshape(-1)


@lru_cache(maxsize=4)
def get_reranker(model_name: str, device: str | None = None,
                 batch_size: int = 32, max_length: int = 512) -> Reranker:
    """Process-wide singleton keyed by (model, device, batch_size, max_length)."""
    return Reranker(model_name, device=device, batch_size=batch_size,
                    max_length=max_length)
