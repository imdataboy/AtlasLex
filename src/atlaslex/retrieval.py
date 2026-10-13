"""Original dense + BM25 → RRF → cross-encoder retrieval pipeline."""

import re

from rank_bm25 import BM25Okapi

from .preparation.chunking import canonical_chunk_id


def simple_tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\sàâäéèêëîïôöùûüç'-]", " ", text.lower()).split()


def build_bm25(chunks):
    if not chunks:
        return None
    return BM25Okapi([simple_tokenize(c["prefixed_text"]) or ["__empty__"] for c in chunks])


def reciprocal_rank_fusion(dense, sparse, k=60):
    scores = {}
    for results in [dense, sparse]:
        seen = set()
        for rank, result in enumerate(results, 1):
            cid = canonical_chunk_id(result["chunk_id"])
            if cid not in seen:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                seen.add(cid)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def dedup_by_article(chunks, top_k):
    """Group retrieved subchunks, retaining their distinct evidence and page spans.

    top_k counts articles (or fallback page sections), not raw overlapping chunks.
    Identical article numbers from different documents remain separate.
    """
    groups = {}
    for chunk in chunks:
        key = (chunk["document_id"], chunk["section_id"])
        if key not in groups:
            if len(groups) == top_k:
                continue
            groups[key] = []
        if not any(c["chunk_id"] == chunk["chunk_id"] for c in groups[key]):
            groups[key].append(chunk)
    result = []
    for members in groups.values():
        combined = dict(members[0])
        ordered = sorted(members, key=lambda c: c["sub_chunk_index"])
        text, previous_index = "", -2
        for chunk in ordered:
            part = chunk["text"]
            overlap = 0
            if chunk["sub_chunk_index"] == previous_index + 1:
                for count in range(min(len(text), len(part)), 0, -1):
                    if text.endswith(part[:count]):
                        overlap = count
                        break
            text += part[overlap:] if overlap else (("\n\n" if text else "") + part)
            previous_index = chunk["sub_chunk_index"]
        combined.update(
            text=text,
            chunk_ids=[c["chunk_id"] for c in ordered],
            page_numbers=sorted({p for c in ordered for p in c["page_numbers"]}),
            footnotes=list(dict.fromkeys(n for c in ordered for n in c["footnotes"])),
        )
        result.append(combined)
    return result


class Retriever:
    def __init__(self, manager):
        self.manager = manager
        self.settings = manager.settings
        self.models = manager.models
        self.collection, self.manifest, self.chunks = manager.load()
        self.lookup = {canonical_chunk_id(c["chunk_id"]): c for c in self.chunks}
        self.bm25 = build_bm25(self.chunks)

    def search(self, query: str, top_k: int | None = None):
        top_k = self.settings.final_top_k if top_k is None else top_k
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if not query.strip() or not self.chunks:
            return []
        self.manager._check_dimensions(self.collection, self.models.dimension)
        dense_k, sparse_k = (
            max(top_k, self.settings.dense_top_k),
            max(top_k, self.settings.sparse_top_k),
        )
        points = self.manager.client.query_points(
            self.collection,
            query=self.models.embed_query(query),
            limit=dense_k,
            with_payload=False,
        ).points
        dense = [{"chunk_id": canonical_chunk_id(p.id), "score": p.score} for p in points]
        scores = self.bm25.get_scores(simple_tokenize(query))
        sparse = [
            {"chunk_id": self.chunks[i]["chunk_id"], "score": float(scores[i])}
            for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:sparse_k]
            if scores[i] > 0
        ]
        ids = reciprocal_rank_fusion(dense, sparse)
        candidates = [self.lookup[cid] for cid in ids if cid in self.lookup]
        ranked = self.models.rerank(query, candidates)
        relevant = [c for c in ranked if c["_rerank_score"] >= self.settings.min_rerank_score]
        return dedup_by_article(relevant, top_k)
