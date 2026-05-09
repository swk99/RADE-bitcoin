"""
btc_qa/memory.py
----------------
Episodic memory for Bitcoin node diagnostic agent.
Stores (state, action, reward) tuples in ChromaDB and retrieves
similar past episodes to augment LLM interpretation context.
"""

import uuid
from datetime import datetime, timezone
import chromadb
from dataclasses import dataclass
from typing import Optional


@dataclass
class Episode:
    mempool_size: float       # MB
    pending_tx: int           # count
    inter_block_time: float   # seconds
    action: str               # probe / skip / escalate
    reward: float             # net reward after lambda penalty
    cost: float               # probe cost
    detected: bool            # anomaly detected?
    step: Optional[int] = None
    timestamp: Optional[float] = None
    episode_id: Optional[str] = None

    def to_vector(self) -> list[float]:
        """Simple feature vector for similarity search."""
        # Backward compatible encoding for legacy action names.
        action_enc = {
            "probe": 1.0,
            "escalate": -1.0,
            "skip": 0.0,
            "buy": 1.0,
            "sell": -1.0,
            "hold": 0.0,
        }
        return [
            self.mempool_size / 120.0,
            self.pending_tx / 8000.0,
            self.inter_block_time / 1200.0,
            action_enc.get(self.action, 0.0),
            max(-1.0, min(1.0, self.reward)),
        ]

    def to_metadata(self) -> dict:
        ts = self.timestamp
        if ts is None:
            ts = datetime.now(timezone.utc).timestamp()
        metadata = {
            "mempool_size": round(self.mempool_size, 2),
            "pending_tx": int(self.pending_tx),
            "inter_block_time": round(self.inter_block_time, 1),
            "action": self.action,
            "reward": round(self.reward, 4),
            "cost": round(self.cost, 4),
            "detected": int(self.detected),
            "timestamp": float(ts),
        }
        if self.step is not None:
            metadata["step"] = int(self.step)
        return metadata

    def to_document(self) -> str:
        return (
            f"State: mempool={self.mempool_size:.1f}MB "
            f"pending={self.pending_tx} tx "
            f"inter_block={self.inter_block_time:.0f}s | "
            f"Action: {self.action} | "
            f"Reward: {self.reward:+.4f} | "
            f"Detected: {self.detected}"
        )


class EpisodicMemory:
    """
    ChromaDB-backed episodic memory.

    Usage:
        mem = EpisodicMemory(persist_dir="./btc_memory")
        mem.add(episode)
        similar = mem.retrieve(query_episode, k=3)
    """

    def __init__(self, persist_dir: str = "./btc_memory", collection: str = "episodes"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, episode: Episode) -> str:
        eid = episode.episode_id or str(uuid.uuid4())
        self.col.add(
            ids=[eid],
            embeddings=[episode.to_vector()],
            documents=[episode.to_document()],
            metadatas=[episode.to_metadata()],
        )
        return eid

    def retrieve(self, query: Episode, k: int = 3) -> list[dict]:
        """Return top-k similar past episodes."""
        n = self.col.count()
        if n == 0:
            return []
        k = min(k, n)
        results = self.col.query(
            query_embeddings=[query.to_vector()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"document": doc, "metadata": meta, "similarity": round(1 - dist, 3)})
        return out

    def retrieve_causal(self, query: Episode, k: int, current_step: int) -> list[dict]:
        """Return top-k episodes from strictly earlier steps (step < current_step)."""
        n = self.col.count()
        if n == 0:
            return []
        try:
            results = self.col.query(
                query_embeddings=[query.to_vector()],
                n_results=min(k, n),
                where={"step": {"$lt": int(current_step)}},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []
        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"document": doc, "metadata": meta, "similarity": round(1 - dist, 3)})
        return out

    @staticmethod
    def rag_belief(similar: list[dict], k: int) -> float:
        """
        Compute RAG belief:
            b_hat = (1/K) * sum_k sim_k * I[detected_k]
        """
        if k <= 0:
            return 0.0
        score = 0.0
        used = similar[:k]
        if not used:
            return 0.0
        for item in used:
            sim = float(item.get("similarity", 0.0))
            detected = int(item.get("metadata", {}).get("detected", 0))
            score += sim * detected
        belief = score / float(max(1, len(used)))
        return max(0.0, min(1.0, belief))

    def count(self) -> int:
        return self.col.count()

    def clear(self):
        self.client.delete_collection("episodes")
        self.col = self.client.get_or_create_collection("episodes")
