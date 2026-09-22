"""
retrieval.py — Grounded hybrid retrieval layer for LexTrace AI.

Retrieval architecture:

    Query
      |
      +--------------------+
      |                    |
      v                    v
    BM25              Semantic Search
    TF-IDF             MiniLM embeddings
      |                    |
      +---------+----------+
                |
                v
        Hybrid score fusion
                |
                v
        Ranked evidence passages

Primary semantic model:
    all-MiniLM-L6-v2

Fallback:
    BM25-lite / TF-IDF retrieval

The public interface used by generation.py is:

    RetrievalStore.add_documents(...)
    RetrievalStore.retrieve(...)
    RetrievalStore.retrieve_for_draft(...)
    RetrievalStore.get_doc_meta(...)
    RetrievalStore.stats(...)
"""

from __future__ import annotations

import math
import re

import numpy as np

from collections import Counter
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Passage:
    """
    A retrievable text chunk from a legal document.
    """

    passage_id: str
    doc_id: str
    doc_type: str
    source_path: str
    text: str

    # Global chunk index within the document.
    chunk_index: int

    # Page containing this chunk.
    page_number: int

    # Character offsets within the page text.
    start_char: int
    end_char: int

    # Estimated OCR/document quality for this evidence.
    source_quality: float = 1.0

@dataclass
class RetrievedEvidence:
    """
    Evidence returned by the retrieval layer.
    """

    passage: Passage
    score: float
    query: str
    retrieval_method: str = "hybrid"

    def citation(self) -> str:
        """
        Human-readable source citation.
        Example:
        [Source: title_deed_001.txt, page 2, chunk 3]
        """
        source_name = self.passage.source_path
        
        # Convert Windows paths to just the filename.
        source_name = source_name.replace("\\", "/").split("/")[-1]
        
        return (
            f"[Source: {source_name}, "
            f"page {self.passage.page_number}, "
            f"chunk {self.passage.chunk_index}]"
        )


# ============================================================================
# TEXT CHUNKING
# ============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[tuple[int, int, str]]:
    """
    Split document text into overlapping paragraph-aware chunks.

    Returns:
        [
            (start_character, end_character, chunk_text),
            ...
        ]
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    paragraphs = re.split(r"\n\s*\n+", text)

    chunks: list[tuple[int, int, str]] = []

    buffer = ""
    buffer_start = 0
    position = 0

    for raw_para in paragraphs:

        para = raw_para.strip()

        if not para:
            position += len(raw_para) + 1
            continue

        # ------------------------------------------------------------
        # Add paragraph to current chunk if it fits.
        # ------------------------------------------------------------
        if buffer and len(buffer) + len(para) + 1 <= chunk_size:

            buffer += "\n" + para

        elif not buffer:

            buffer_start = position
            buffer = para

        else:

            # --------------------------------------------------------
            # Flush current chunk.
            # --------------------------------------------------------
            chunks.append(
                (
                    buffer_start,
                    buffer_start + len(buffer),
                    buffer,
                )
            )

            # --------------------------------------------------------
            # Preserve overlap between chunks.
            # --------------------------------------------------------
            overlap_text = (
                buffer[-overlap:]
                if overlap > 0
                else ""
            )

            buffer_start = (
                buffer_start
                + len(buffer)
                - len(overlap_text)
            )

            buffer = (
                overlap_text
                + "\n"
                + para
            )

        position += len(raw_para) + 1

    # ------------------------------------------------------------
    # Flush final chunk.
    # ------------------------------------------------------------
    if buffer.strip():

        chunks.append(
            (
                buffer_start,
                buffer_start + len(buffer),
                buffer,
            )
        )

    return chunks

def chunk_document_pages(
    page_texts: list[str],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[tuple[int, int, int, str]]:
    """
    Chunk a document while preserving page numbers.

    Returns:

        (
            page_number,
            start_char,
            end_char,
            chunk_text
        )
    """

    results = []

    for page_index, page_text in enumerate(
        page_texts,
        start=1,
    ):

        if not page_text or not page_text.strip():
            continue

        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for (
            start_char,
            end_char,
            text,
        ) in page_chunks:

            results.append(
                (
                    page_index,
                    start_char,
                    end_char,
                    text,
                )
            )

    return results


# ============================================================================
# TOKENIZATION
# ============================================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "as",
    "from",
    "and",
    "or",
    "but",
    "not",
    "this",
    "that",
    "it",
    "its",
    "their",
    "they",
    "he",
    "she",
    "we",
    "i",
    "you",
    "my",
    "your",
    "our",
    "his",
    "her",
    "which",
    "who",
    "whom",
    "what",
    "when",
    "where",
    "how",
    "all",
    "any",
    "each",
    "if",
    "then",
    "so",
    "than",
    "into",
    "about",
    "such",
    "no",
    "up",
    "out",
    "more",
    "also",
    "only",
    "just",
    "after",
    "before",
    "during",
}


def tokenize(text: str) -> list[str]:
    """
    Lightweight legal-text tokenizer.

    Keeps:
        alphabetic tokens
        numeric tokens
        alphanumeric identifiers

    This is intentionally dependency-free for the BM25 fallback.
    """

    if not text:
        return []

    tokens = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
        and len(token) > 1
    ]


# ============================================================================
# BM25 / TF-IDF RETRIEVAL
# ============================================================================

class TFIDFIndex:
    """
    Lightweight BM25-style retrieval implementation.

    This is the dependency-free fallback when semantic retrieval
    cannot be loaded.
    """

    def __init__(self):
        self.passages: list[Passage] = []

        self.doc_freqs: Counter = Counter()

        self.passage_tfs: list[Counter] = []

        self.num_passages: int = 0

        self.average_length: float = 0.0

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def add_passages(
        self,
        passages: list[Passage],
    ) -> None:

        # Reset index so repeated calls don't duplicate passages.
        self.passages = []
        self.doc_freqs = Counter()
        self.passage_tfs = []

        for passage in passages:

            tokens = tokenize(passage.text)

            tf = Counter(tokens)

            self.passages.append(passage)

            self.passage_tfs.append(tf)

            self.doc_freqs.update(
                set(tokens)
            )

        self.num_passages = len(
            self.passages
        )

        if self.num_passages:

            total_tokens = sum(
                sum(tf.values())
                for tf in self.passage_tfs
            )

            self.average_length = (
                total_tokens
                / self.num_passages
            )

        else:

            self.average_length = 0.0

    # ------------------------------------------------------------------
    # IDF
    # ------------------------------------------------------------------

    def _idf(
        self,
        term: str,
    ) -> float:

        document_frequency = (
            self.doc_freqs.get(term, 0)
        )

        if document_frequency == 0:
            return 0.0

        return math.log(
            (
                self.num_passages + 1
            )
            / (
                document_frequency + 1
            )
        ) + 1.0

    # ------------------------------------------------------------------
    # BM25 score
    # ------------------------------------------------------------------

    def _score(
        self,
        query_tokens: list[str],
        passage_tf: Counter,
        passage_len: int,
    ) -> float:

        if not query_tokens:
            return 0.0

        if self.average_length <= 0:
            return 0.0

        k1 = 1.5
        b = 0.75

        score = 0.0

        for term in query_tokens:

            tf = passage_tf.get(
                term,
                0,
            )

            if tf == 0:
                continue

            idf = self._idf(term)

            numerator = (
                tf * (k1 + 1.0)
            )

            denominator = (
                tf
                + k1
                * (
                    1.0
                    - b
                    + b
                    * passage_len
                    / self.average_length
                )
            )

            score += (
                idf
                * numerator
                / denominator
            )

        return float(score)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Passage, float]]:

        if not self.passages:
            return []

        if top_k <= 0:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scored: list[
            tuple[Passage, float]
        ] = []

        for passage, tf in zip(
            self.passages,
            self.passage_tfs,
        ):

            passage_length = sum(
                tf.values()
            )

            score = self._score(
                query_tokens,
                tf,
                passage_length,
            )

            if score > 0:
                scored.append(
                    (
                        passage,
                        score,
                    )
                )

        # Deterministic ranking:
        # first by score, then passage ID.
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].passage_id,
            )
        )

        return scored[:top_k]


# ============================================================================
# SEMANTIC RETRIEVAL
# ============================================================================

class SemanticIndex:
    """
    Dense-vector retrieval using sentence-transformers.

    Model:
        all-MiniLM-L6-v2

    Embeddings are L2-normalized, so cosine similarity is
    computed using a simple dot product.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):

        self.passages: list[Passage] = []

        self.embeddings: Optional[
            np.ndarray
        ] = None

        self._model = None

        self._available = False

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

            self._model = SentenceTransformer(
                self.MODEL_NAME
            )

            self._available = True

        except ImportError:

            print(
                "[retrieval] sentence-transformers "
                "not installed — using BM25 fallback"
            )

            print(
                "           Install with: "
                "pip install sentence-transformers"
            )

        except Exception as exc:

            print(
                "[retrieval] Could not load semantic "
                f"model ({exc}) — using BM25 fallback"
            )

            self._model = None
            self._available = False

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return (
            self._available
            and self._model is not None
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def add_passages(
        self,
        passages: list[Passage],
    ) -> None:

        self.passages = list(passages)

        if not self.available:
            self.embeddings = None
            return

        if not self.passages:
            self.embeddings = None
            return

        texts = [
            passage.text
            for passage in self.passages
        ]

        self.embeddings = np.asarray(
            self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Passage, float]]:

        if not self.available:
            return []

        if self.embeddings is None:
            return []

        if not self.passages:
            return []

        if top_k <= 0:
            return []

        if not query or not query.strip():
            return []

        query_vector = np.asarray(
            self._model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        )

        # Since both vectors are normalized:
        # cosine similarity = dot product.
        scores = (
            self.embeddings
            @ query_vector
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results: list[
            tuple[Passage, float]
        ] = []

        for index in top_indices:

            score = float(
                scores[index]
            )

            # Ignore extremely weak semantic matches.
            if score > 0.10:

                results.append(
                    (
                        self.passages[index],
                        score,
                    )
                )

        return results


# ============================================================================
# HYBRID RETRIEVAL
# ============================================================================

class HybridIndex:
    """
    Combines BM25 and semantic retrieval.

    Final score:

        0.6 * semantic_score
        +
        0.4 * normalized_bm25_score

    Semantic retrieval handles:
        - paraphrases
        - conceptual similarity
        - natural-language queries

    BM25 handles:
        - exact names
        - case references
        - monetary values
        - clause/section numbers
        - legal terminology
    """

    SEMANTIC_WEIGHT = 0.60
    BM25_WEIGHT = 0.40

    def __init__(self):

        self.bm25 = TFIDFIndex()

        self.semantic = SemanticIndex()

        self.passages: list[Passage] = []

    # ------------------------------------------------------------------
    # Retrieval method
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:

        if self.semantic.available:
            return "hybrid"

        return "bm25"

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def add_passages(
        self,
        passages: list[Passage],
    ) -> None:

        self.passages = list(passages)

        self.bm25.add_passages(
            self.passages
        )

        if self.semantic.available:

            self.semantic.add_passages(
                self.passages
            )

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[
        tuple[Passage, float, str]
    ]:

        if top_k <= 0:
            return []

        # ==============================================================
        # BM25 FALLBACK
        # ==============================================================

        if not self.semantic.available:

            bm25_results = (
                self.bm25.retrieve(
                    query,
                    top_k=top_k,
                )
            )

            return [
                (
                    passage,
                    score,
                    "bm25",
                )
                for passage, score
                in bm25_results
            ]

        # ==============================================================
        # GET CANDIDATES FROM BOTH RETRIEVERS
        # ==============================================================

        bm25_raw = self.bm25.retrieve(
            query,
            top_k=top_k * 2,
        )

        semantic_raw = (
            self.semantic.retrieve(
                query,
                top_k=top_k * 2,
            )
        )

        # ==============================================================
        # INDEX BY passage_id
        #
        # IMPORTANT:
        # Passage objects are NOT used as dictionary keys.
        # This fixes:
        #
        # TypeError: unhashable type: 'Passage'
        # ==============================================================

        bm25_results: dict[
            str,
            tuple[Passage, float]
        ] = {
            passage.passage_id: (
                passage,
                score,
            )
            for passage, score
            in bm25_raw
        }

        semantic_results: dict[
            str,
            tuple[Passage, float]
        ] = {
            passage.passage_id: (
                passage,
                score,
            )
            for passage, score
            in semantic_raw
        }

        # ==============================================================
        # NORMALIZE BM25
        # ==============================================================

        if bm25_results:

            max_bm25 = max(
                score
                for _, score
                in bm25_results.values()
            )

            if max_bm25 <= 0:
                max_bm25 = 1.0

            bm25_normalized = {
                passage_id: (
                    score / max_bm25
                )
                for passage_id, (
                    _,
                    score,
                ) in bm25_results.items()
            }

        else:

            bm25_normalized = {}

        # ==============================================================
        # COMBINE CANDIDATES
        #
        # dict.fromkeys preserves deterministic insertion order.
        # ==============================================================

        candidate_ids = list(
            dict.fromkeys(
                list(bm25_results.keys())
                + list(semantic_results.keys())
            )
        )

        if not candidate_ids:
            return []

        # ==============================================================
        # CALCULATE HYBRID SCORE
        # ==============================================================

        scored: list[
            tuple[Passage, float, str]
        ] = []

        for passage_id in candidate_ids:

            # ----------------------------------------------------------
            # Get Passage object
            # ----------------------------------------------------------

            if passage_id in semantic_results:

                passage = (
                    semantic_results[
                        passage_id
                    ][0]
                )

                semantic_score = (
                    semantic_results[
                        passage_id
                    ][1]
                )

            else:

                passage = (
                    bm25_results[
                        passage_id
                    ][0]
                )

                semantic_score = 0.0

            # ----------------------------------------------------------
            # Get normalized BM25 score
            # ----------------------------------------------------------

            bm25_score = (
                bm25_normalized.get(
                    passage_id,
                    0.0,
                )
            )

            # ----------------------------------------------------------
            # Weighted hybrid score
            # ----------------------------------------------------------

            combined_score = (
                self.SEMANTIC_WEIGHT
                * semantic_score
                +
                self.BM25_WEIGHT
                * bm25_score
            )

            scored.append(
                (
                    passage,
                    float(combined_score),
                    "hybrid",
                )
            )

        # ==============================================================
        # SORT
        # ==============================================================

        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].passage_id,
            )
        )

        return scored[:top_k]


# ============================================================================
# RETRIEVAL STORE
# ============================================================================

class RetrievalStore:
    """
    Public retrieval API used by LexTrace AI.

    Handles:

        documents
            ↓
        chunking
            ↓
        BM25 indexing
            +
        semantic indexing
            ↓
        hybrid retrieval
            ↓
        RetrievedEvidence
    """

    def __init__(self):

        self.index = HybridIndex()

        self._doc_registry: dict[
            str,
            dict,
        ] = {}

        self._all_passages: list[
            Passage
        ] = []

    # ------------------------------------------------------------------
    # Add one document
    # ------------------------------------------------------------------

    def add_document(
        self,
        extracted_doc,
    ) -> int:

        chunks = chunk_document_pages(
            extracted_doc.page_texts
        )
        
        passages = []
        
        for index, (
            page_number,
            start,
            end,
            text,
        ) in enumerate(chunks):
            
            passage_id = (
                f"{extracted_doc.doc_id}"
                f"_page{page_number}"
                f"_chunk{index}"
            )
            
            quality_score = (
                extracted_doc.confidence_flags.get(
                "quality_score",
                 100,
                )
            )
            
            source_quality = (
                max(
                    0.0,
                    min(
                        1.0,
                        quality_score / 100.0,
                        ),
                    )
                )
            
            passages.append(
                Passage(
                    passage_id=passage_id,
                    doc_id=extracted_doc.doc_id,
                    doc_type=extracted_doc.doc_type,
                    source_path=extracted_doc.source_path,
                    text=text,
                    chunk_index=index,
                    page_number=page_number,
                    start_char=start,
                    end_char=end,
                    source_quality=source_quality,
                )
            )

        self._all_passages.extend(
            passages
        )

        # Rebuild the complete index.
        self.index = HybridIndex()

        self.index.add_passages(
            self._all_passages
        )

        self._doc_registry[
            extracted_doc.doc_id
        ] = {
            "doc_type": extracted_doc.doc_type,
            "source_path": extracted_doc.source_path,
            "structured_fields": (
                extracted_doc.structured_fields
            ),
            "confidence_flags": (
                extracted_doc.confidence_flags
            ),
            "num_chunks": len(passages),
        }

        return len(passages)

    # ------------------------------------------------------------------
    # Add multiple documents
    # ------------------------------------------------------------------

    def add_documents(
        self,
        extracted_docs: list,
    ) -> dict:

        summary: dict[
            str,
            int,
        ] = {}

        all_new_passages: list[
            Passage
        ] = []

        for doc in extracted_docs:

            chunks = chunk_document_pages(
                doc.page_texts
            )
            
            passages = []
            
            for index, (
                page_number,
                start,
                end,
                text,
            ) in enumerate(chunks):
                passage_id = (
                    f"{doc.doc_id}"
                    f"_page{page_number}"
                    f"_chunk{index}"
                )

                # Convert document quality score into
                # a 0–1 evidence confidence.
                quality_score = (
                    doc.confidence_flags.get(
                        "quality_score",
                        100,
                    )
                )
                source_quality = (
                    max(
                        0.0,
                        min(
                            1.0,
                            quality_score / 100.0,
                        ),
                    )
                )

                passages.append(
                    Passage(
                        passage_id=passage_id,
                        doc_id=doc.doc_id,
                        doc_type=doc.doc_type,
                        source_path=doc.source_path,
                        text=text,
                        chunk_index=index,
                        page_number=page_number,
                        start_char=start,
                        end_char=end,
                        source_quality=source_quality,
                    )
                )
            all_new_passages.extend(
                passages
            )

            self._doc_registry[
                doc.doc_id
            ] = {
                "doc_type": doc.doc_type,
                "source_path": doc.source_path,
                "structured_fields": (
                    doc.structured_fields
                ),
                "confidence_flags": (
                    doc.confidence_flags
                ),
                "num_chunks": len(
                    passages
                ),
            }

            summary[
                doc.doc_id
            ] = len(passages)

        self._all_passages.extend(
            all_new_passages
        )

        # Build both retrieval indexes
        # once for efficiency.
        self.index = HybridIndex()

        self.index.add_passages(
            self._all_passages
        )

        return summary

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: Optional[str] = None,
    ) -> list[RetrievedEvidence]:

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        # Retrieve extra candidates because
        # doc_type filtering may remove some.
        raw = self.index.retrieve(
            query,
            top_k=top_k * 2,
        )

        if not raw:
            return []

        evidence: list[
            RetrievedEvidence
        ] = []

        for (
            passage,
            score,
            method,
        ) in raw:

            # ----------------------------------------------------------
            # Optional document-type filtering
            # ----------------------------------------------------------

            if (
                doc_type_filter
                and passage.doc_type
                != doc_type_filter
            ):
                continue

            evidence.append(
                RetrievedEvidence(
                    passage=passage,
                    score=float(score),
                    query=query,
                    retrieval_method=method,
                )
            )

            if len(evidence) >= top_k:
                break

        return evidence

    # ------------------------------------------------------------------
    # Retrieval for generation
    # ------------------------------------------------------------------

    def retrieve_for_draft(
        self,
        draft_sections: dict[str, str],
    ) -> dict[
        str,
        list[RetrievedEvidence],
    ]:

        if not draft_sections:
            return {}

        results: dict[
            str,
            list[RetrievedEvidence],
        ] = {}

        for section, query in (
            draft_sections.items()
        ):

            results[section] = (
                self.retrieve(
                    query,
                    top_k=3,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Document metadata
    # ------------------------------------------------------------------

    def get_doc_meta(
        self,
        doc_id: str,
    ) -> Optional[dict]:

        return self._doc_registry.get(
            doc_id
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:

        return {
            "total_passages": len(
                self._all_passages
            ),

            "total_documents": len(
                self._doc_registry
            ),

            "doc_types": sorted(
                {
                    metadata["doc_type"]
                    for metadata
                    in self._doc_registry.values()
                }
            ),

            "retrieval_method": (
                self.index.method
            ),

            "semantic_available": (
                self.index.semantic.available
            ),

            "semantic_model": (
                self.index.semantic.MODEL_NAME
                if self.index.semantic.available
                else None
            ),

            "semantic_weight": (
                self.index.SEMANTIC_WEIGHT
            ),

            "bm25_weight": (
                self.index.BM25_WEIGHT
            ),
        }


# ============================================================================
# DIRECT MODULE TEST
# ============================================================================

if __name__ == "__main__":

    import sys
    from pathlib import Path

    # Allow imports when executing this file directly.
    project_root = (
        Path(__file__).resolve().parent.parent
    )

    sys.path.insert(
        0,
        str(project_root),
    )

    from ingestion.ingestion import (
        ingest_directory,
    )

    samples_dir = (
        project_root.parent
        / "samples"
    )

    print("=" * 70)
    print("LexTrace AI — Retrieval Test")
    print("=" * 70)

    print(
        f"\nSamples directory: "
        f"{samples_dir}"
    )

    # ------------------------------------------------------------
    # Ingest documents
    # ------------------------------------------------------------

    documents = ingest_directory(
        str(samples_dir)
    )

    print(
        f"\nLoaded {len(documents)} "
        f"document(s)"
    )

    # ------------------------------------------------------------
    # Build retrieval store
    # ------------------------------------------------------------

    store = RetrievalStore()

    summary = store.add_documents(
        documents
    )

    print(
        f"\nIndexed passages: "
        f"{summary}"
    )

    # ------------------------------------------------------------
    # Print retrieval stats
    # ------------------------------------------------------------

    print("\nRetrieval statistics:")

    stats = store.stats()

    for key, value in stats.items():

        print(
            f"  {key}: {value}"
        )

    # ------------------------------------------------------------
    # Test query
    # ------------------------------------------------------------

    query = (
        "outstanding dues "
        "rent payment tenant"
    )

    print(
        f"\nTest query: {query}"
    )

    results = store.retrieve(
        query,
        top_k=5,
    )

    if not results:

        print(
            "\nNo retrieval results found."
        )

    else:

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{index}] "
                f"score={result.score:.4f} "
                f"method={result.retrieval_method}"
            )

            print(
                f"Document: "
                f"{result.passage.doc_id}"
            )
            
            print(
                f"Page: "
                f"{result.passage.page_number}"
            )
            
            print(
                f"Chunk: "
                f"{result.passage.chunk_index}"
                )
            
            print(
                f"Source quality: "
                f"{result.passage.source_quality:.2f}"
                )

            print(
                f"Citation: "
                f"{result.citation()}"
            )

            print(
                "Text:"
            )

            print(
                result.passage.text[:500]
            )

    print(
        "\n" + "=" * 70
    )
    print(
        "Retrieval test complete."
    )
    print(
        "=" * 70
    )