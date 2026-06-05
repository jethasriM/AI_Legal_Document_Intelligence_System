"""
retrieval.py — Grounded retrieval layer.

Two retrieval backends, automatically selected:

  1. SemanticIndex (preferred) — sentence-transformers + cosine similarity.
     Understands paraphrases: "outstanding consideration" matches "how much is owed".
     Requires: pip install sentence-transformers
     Model: all-MiniLM-L6-v2 (90MB, fast, strong for legal text)

  2. TFIDFIndex (fallback) — custom BM25-lite, zero dependencies.
     Used automatically if sentence-transformers is unavailable or the model
     can't be downloaded (e.g. no internet / sandbox environment).

HybridIndex combines both scores (weighted 0.6 semantic + 0.4 BM25) for the
best of both worlds when both are available.

The retrieval store exposes a single interface regardless of which backend runs.
"""

import re
import math
import numpy as np
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    passage_id: str
    doc_id: str
    doc_type: str
    source_path: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int


@dataclass
class RetrievedEvidence:
    passage: Passage
    score: float
    query: str
    retrieval_method: str = "hybrid"   # "semantic" | "bm25" | "hybrid"

    def citation(self) -> str:
        return f"[Source: {self.passage.doc_id}, chunk {self.passage.chunk_index}]"


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[tuple[int, int, str]]:
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    buffer = ""
    buffer_start = 0
    pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            pos += len(para) + 2
            continue

        if len(buffer) + len(para) + 1 <= chunk_size:
            if buffer:
                buffer += "\n" + para
            else:
                buffer_start = pos
                buffer = para
        else:
            if buffer:
                chunks.append((buffer_start, buffer_start + len(buffer), buffer))
                overlap_text = buffer[-overlap:] if len(buffer) > overlap else buffer
                buffer_start = buffer_start + len(buffer) - len(overlap_text)
                buffer = overlap_text + "\n" + para
            else:
                buffer_start = pos
                buffer = para
        pos += len(para) + 2

    if buffer.strip():
        chunks.append((buffer_start, buffer_start + len(buffer), buffer))

    return chunks


# ---------------------------------------------------------------------------
# BM25-lite TF-IDF index (zero dependency fallback)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being","have",
        "has","had","do","does","did","will","would","could","should","may",
        "might","shall","to","of","in","on","at","by","for","with","as",
        "from","and","or","but","not","this","that","it","its","their",
        "they","he","she","we","i","you","my","your","our","his","her",
        "which","who","whom","what","when","where","how","all","any","each",
        "if","then","so","than","into","about","such","no","up","out",
        "more","also","only","just","been","after","before","during"
    }
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class TFIDFIndex:
    def __init__(self):
        self.passages: list[Passage] = []
        self.doc_freqs: Counter = Counter()
        self.passage_tfs: list[Counter] = []
        self.num_passages: int = 0

    def add_passages(self, passages: list[Passage]):
        for p in passages:
            tokens = tokenize(p.text)
            tf = Counter(tokens)
            self.passage_tfs.append(tf)
            self.doc_freqs.update(set(tokens))
            self.passages.append(p)
        self.num_passages = len(self.passages)

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.num_passages + 1) / (df + 1)) + 1

    def _score(self, query_tokens: list[str], passage_tf: Counter, passage_len: int) -> float:
        k1, b = 1.5, 0.75
        avg_len = sum(sum(tf.values()) for tf in self.passage_tfs) / max(self.num_passages, 1)
        score = 0.0
        for term in query_tokens:
            tf = passage_tf.get(term, 0)
            idf = self._idf(term)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * passage_len / max(avg_len, 1))
            score += idf * (numerator / denominator)
        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Passage, float]]:
        if not self.passages:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored = []
        for passage, tf in zip(self.passages, self.passage_tfs):
            p_len = sum(tf.values())
            score = self._score(query_tokens, tf, p_len)
            scored.append((passage, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, s) for p, s in scored[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Semantic index — sentence-transformers + cosine similarity
# ---------------------------------------------------------------------------

class SemanticIndex:
    """
    Dense vector retrieval using sentence-transformers.

    Uses all-MiniLM-L6-v2:
      - 90MB model, runs on CPU in ~50ms per query
      - Strong semantic understanding for legal text
      - Handles paraphrases that BM25 misses entirely:
          "how much was paid" ↔ "outstanding consideration amount"
          "who owns the property" ↔ "vendor purchaser parties"

    Falls back gracefully to None if the model can't be loaded.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.passages: list[Passage] = []
        self.embeddings: Optional[np.ndarray] = None
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            self._available = True
        except ImportError:
            print("[retrieval] sentence-transformers not installed — using BM25 fallback")
            print("           Install with: pip install sentence-transformers")
        except Exception as e:
            print(f"[retrieval] Could not load semantic model ({e}) — using BM25 fallback")

    @property
    def available(self) -> bool:
        return self._available and self._model is not None

    def add_passages(self, passages: list[Passage]):
        if not self.available:
            return
        self.passages = passages
        texts = [p.text for p in passages]
        # Encode all passages in one batch — efficient
        self.embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Passage, float]]:
        if not self.available or self.embeddings is None or not self.passages:
            return []

        query_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Cosine similarity via dot product (vectors are L2-normalized)
        scores = self.embeddings @ query_vec   # shape: (n_passages,)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.1:   # threshold — ignore very weak matches
                results.append((self.passages[idx], score))

        return results


# ---------------------------------------------------------------------------
# Hybrid index — combines BM25 + semantic scores
# ---------------------------------------------------------------------------

class HybridIndex:
    """
    Combines BM25 and semantic scores with weighted fusion.

    Score = 0.6 * semantic_score + 0.4 * bm25_score (normalised)

    Why hybrid?
    - BM25 is precise for exact legal terms (clause numbers, section references)
    - Semantic catches paraphrases and conceptual queries
    - Together they cover more ground than either alone
    """

    SEMANTIC_WEIGHT = 0.6
    BM25_WEIGHT = 0.4

    def __init__(self):
        self.bm25 = TFIDFIndex()
        self.semantic = SemanticIndex()
        self.passages: list[Passage] = []

    @property
    def method(self) -> str:
        if self.semantic.available:
            return "hybrid"
        return "bm25"

    def add_passages(self, passages: list[Passage]):
        self.passages = passages
        self.bm25.add_passages(passages)
        if self.semantic.available:
            self.semantic.add_passages(passages)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Passage, float, str]]:
        """Returns list of (passage, score, method)."""

        if not self.semantic.available:
            # Pure BM25
            results = self.bm25.retrieve(query, top_k=top_k)
            return [(p, s, "bm25") for p, s in results]

        # Get scores from both
        bm25_results = dict(self.bm25.retrieve(query, top_k=top_k * 2))
        semantic_results = dict(self.semantic.retrieve(query, top_k=top_k * 2))

        # Normalize BM25 scores to [0, 1]
        if bm25_results:
            max_bm25 = max(bm25_results.values()) or 1.0
            bm25_norm = {p: s / max_bm25 for p, s in bm25_results.items()}
        else:
            bm25_norm = {}

        # Collect all candidate passages
        all_passages = set(list(bm25_results.keys()) + list(semantic_results.keys()))

        scored = []
        for passage in all_passages:
            sem_score = semantic_results.get(passage, 0.0)
            bm25_score = bm25_norm.get(passage, 0.0)
            combined = self.SEMANTIC_WEIGHT * sem_score + self.BM25_WEIGHT * bm25_score
            scored.append((passage, combined, "hybrid"))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Retrieval store — public API
# ---------------------------------------------------------------------------

class RetrievalStore:
    """
    Main interface for the generation layer.

    Wraps HybridIndex (semantic + BM25) with automatic fallback to pure BM25
    if sentence-transformers is unavailable.
    """

    def __init__(self):
        self.index = HybridIndex()
        self._doc_registry: dict[str, dict] = {}
        self._all_passages: list[Passage] = []

    def add_document(self, extracted_doc) -> int:
        chunks = chunk_text(extracted_doc.cleaned_text)
        passages = []
        for i, (start, end, text) in enumerate(chunks):
            pid = f"{extracted_doc.doc_id}_chunk{i}"
            passages.append(Passage(
                passage_id=pid,
                doc_id=extracted_doc.doc_id,
                doc_type=extracted_doc.doc_type,
                source_path=extracted_doc.source_path,
                text=text,
                chunk_index=i,
                start_char=start,
                end_char=end,
            ))

        self._all_passages.extend(passages)
        # Re-index all passages together (semantic model encodes in batch)
        self.index = HybridIndex()
        self.index.add_passages(self._all_passages)

        self._doc_registry[extracted_doc.doc_id] = {
            "doc_type": extracted_doc.doc_type,
            "source_path": extracted_doc.source_path,
            "structured_fields": extracted_doc.structured_fields,
            "confidence_flags": extracted_doc.confidence_flags,
            "num_chunks": len(passages),
        }
        return len(passages)

    def add_documents(self, extracted_docs: list) -> dict:
        # Collect all passages first, then index in one batch
        summary = {}
        all_new_passages = []

        for doc in extracted_docs:
            chunks = chunk_text(doc.cleaned_text)
            passages = []
            for i, (start, end, text) in enumerate(chunks):
                pid = f"{doc.doc_id}_chunk{i}"
                passages.append(Passage(
                    passage_id=pid,
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    source_path=doc.source_path,
                    text=text,
                    chunk_index=i,
                    start_char=start,
                    end_char=end,
                ))
            all_new_passages.extend(passages)
            self._doc_registry[doc.doc_id] = {
                "doc_type": doc.doc_type,
                "source_path": doc.source_path,
                "structured_fields": doc.structured_fields,
                "confidence_flags": doc.confidence_flags,
                "num_chunks": len(passages),
            }
            summary[doc.doc_id] = len(passages)

        self._all_passages.extend(all_new_passages)
        # Single batch index — efficient for semantic model
        self.index = HybridIndex()
        self.index.add_passages(self._all_passages)
        return summary

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> list[RetrievedEvidence]:

        raw = self.index.retrieve(query, top_k=top_k * 2)
        evidence = []
        for passage, score, method in raw:
            if doc_type_filter and passage.doc_type != doc_type_filter:
                continue
            evidence.append(RetrievedEvidence(
                passage=passage,
                score=score,
                query=query,
                retrieval_method=method,
            ))
            if len(evidence) == top_k:
                break
        return evidence

    def retrieve_for_draft(self, draft_sections: dict[str, str]) -> dict[str, list[RetrievedEvidence]]:
        return {
            section: self.retrieve(query, top_k=3)
            for section, query in draft_sections.items()
        }

    def get_doc_meta(self, doc_id: str) -> Optional[dict]:
        return self._doc_registry.get(doc_id)

    def stats(self) -> dict:
        return {
            "total_passages": len(self._all_passages),
            "total_documents": len(self._doc_registry),
            "doc_types": list({m["doc_type"] for m in self._doc_registry.values()}),
            "retrieval_method": self.index.method,
            "semantic_available": self.index.semantic.available,
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from ingestion.ingestion import ingest_directory

    samples_dir = __import__("pathlib").Path(__file__).parent.parent.parent / "samples"
    docs = ingest_directory(str(samples_dir))
    store = RetrievalStore()
    summary = store.add_documents(docs)
    print(f"Indexed: {summary}")
    print(f"Stats: {store.stats()}")

    results = store.retrieve("outstanding dues rent payment tenant")
    for r in results:
        print(f"\n[{r.score:.3f}] {r.retrieval_method} | {r.passage.doc_id} chunk {r.passage.chunk_index}")
        print(r.passage.text[:200])
