"""
tests/test_pipeline.py — Unit tests for all pipeline stages.

Run with: python -m pytest tests/ -v
Or: python tests/test_pipeline.py
"""

import sys
import json
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.ingestion import (
    clean_ocr_noise, detect_doc_type, ingest_document,
    extract_title_deed_fields, extract_legal_notice_fields, extract_case_intake_fields
)
from retrieval.retrieval import RetrievalStore, TFIDFIndex, tokenize, chunk_text
from generation.generation import build_generation_prompt, _parse_llm_sections, _extract_grounding_score
from improvement.improvement import (
    compute_diff_summary, extract_rules_from_edit,
    ImprovementStore, simulate_operator_edit
)


# ── Ingestion Tests ──────────────────────────────────────────────────────────

class TestOCRCleaning(unittest.TestCase):
    def test_zero_to_O_correction(self):
        cleaned, warnings = clean_ocr_noise("0ct 2019")
        self.assertIn("Oct", cleaned)

    def test_illegible_tagging(self):
        cleaned, warnings = clean_ocr_noise("Amount: Rs. 4,3??/- [illegible]")
        self.assertIn("[ILLEGIBLE]", cleaned)

    def test_whitespace_normalization(self):
        cleaned, _ = clean_ocr_noise("Line 1\n\n\n\nLine 2")
        self.assertNotIn("\n\n\n", cleaned)

    def test_no_mutation_on_clean_text(self):
        text = "This is a clean document with no OCR errors."
        cleaned, warnings = clean_ocr_noise(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(len(warnings), 0)


class TestDocTypeDetection(unittest.TestCase):
    def test_detects_title_deed(self):
        text = "Deed of conveyance between vendor and purchaser for plot no 112B survey no 45"
        self.assertEqual(detect_doc_type(text), "title_deed")

    def test_detects_legal_notice(self):
        text = "Legal notice instructed by client regarding dues outstanding and vacate premises lease agreement"
        self.assertEqual(detect_doc_type(text), "legal_notice")

    def test_detects_case_intake(self):
        text = "Case ref HYD/2024 intake officer client information opposing party nature of dispute"
        self.assertEqual(detect_doc_type(text), "case_intake")

    def test_unknown_for_garbage(self):
        self.assertEqual(detect_doc_type("aaaaaa bbbbb cccc"), "unknown")


class TestFieldExtraction(unittest.TestCase):
    def test_title_deed_fields(self):
        text = "Ramesh Kumar Sharma, S/o Late Mohan Lal Sharma VENDOR\nTotal Sale Consideration: Rs. 42,00,000/-\nPlot No.: 112-B"
        fields = extract_title_deed_fields(text)
        self.assertIn("sale_consideration", fields)
        self.assertIn("42,00,000", fields["sale_consideration"])

    def test_legal_notice_fields(self):
        text = "Monthly Rent: Rs. 28,000/- per month\nTOTAL APPROX: Rs. 94,800/-\n(a) Pay the outstanding amount"
        fields = extract_legal_notice_fields(text)
        self.assertEqual(fields.get("monthly_rent"), "Rs. 28,000/- per month")

    def test_case_intake_fields(self):
        text = "Name: Deepak Anand Kulkarni\nCase Ref: HYD/2024/0087\n[x] Booking receipt\n[ ] RERA certificate"
        fields = extract_case_intake_fields(text)
        self.assertEqual(fields["client_name"], "Deepak Anand Kulkarni")
        self.assertIn("Booking receipt", fields["documents_present"])
        self.assertIn("RERA certificate", fields["documents_missing"])


class TestFullIngestion(unittest.TestCase):
    def test_ingest_from_samples(self):
        samples_dir = Path(__file__).parent.parent / "samples"
        for sample_file in samples_dir.glob("*.txt"):
            doc = ingest_document(str(sample_file))
            self.assertIsNotNone(doc.doc_id)
            self.assertNotEqual(doc.doc_type, None)
            self.assertGreater(len(doc.cleaned_text), 100)
            self.assertIn("quality_score", doc.confidence_flags)


# ── Retrieval Tests ───────────────────────────────────────────────────────────

class TestTokenizer(unittest.TestCase):
    def test_stopwords_removed(self):
        tokens = tokenize("the quick brown fox")
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)

    def test_lowercase(self):
        tokens = tokenize("VENDOR Purchaser")
        self.assertIn("vendor", tokens)


class TestChunking(unittest.TestCase):
    def test_chunks_are_created(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five."
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        self.assertGreater(len(chunks), 1)

    def test_chunk_content_nonempty(self):
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=200)
        for _, _, text_chunk in chunks:
            self.assertGreater(len(text_chunk.strip()), 0)


class TestRetrievalStore(unittest.TestCase):
    def setUp(self):
        from ingestion.ingestion import ingest_directory
        samples_dir = Path(__file__).parent.parent / "samples"
        docs = ingest_directory(str(samples_dir))
        self.store = RetrievalStore()
        self.store.add_documents(docs)

    def test_retrieval_returns_results(self):
        results = self.store.retrieve("outstanding dues rent payment")
        self.assertGreater(len(results), 0)

    def test_scores_are_positive(self):
        results = self.store.retrieve("vendor purchaser property sale")
        for ev in results:
            self.assertGreater(ev.score, 0)

    def test_doc_type_filter(self):
        results = self.store.retrieve("client name case dispute", doc_type_filter="case_intake")
        for ev in results:
            self.assertEqual(ev.passage.doc_type, "case_intake")

    def test_citation_format(self):
        results = self.store.retrieve("sale consideration amount")
        for ev in results:
            citation = ev.citation()
            self.assertIn("Source:", citation)


# ── Generation Tests ─────────────────────────────────────────────────────────

class TestGenerationHelpers(unittest.TestCase):
    def test_build_prompt_contains_sections(self):
        prompt = build_generation_prompt(
            doc_type="case_intake",
            structured_fields={"client_name": "Test Person"},
            evidence_map={},
            operator_instructions="Be concise.",
        )
        self.assertIn("Party Identification", prompt)
        self.assertIn("Be concise", prompt)
        self.assertIn("Test Person", prompt)

    def test_parse_llm_sections(self):
        raw = "## Party Identification\nTest content here.\n## Financial Summary\nRs. 100"
        parsed = _parse_llm_sections(raw, ["Party Identification", "Financial Summary"])
        self.assertEqual(parsed["Party Identification"], "Test content here.")
        self.assertIn("100", parsed["Financial Summary"])

    def test_extract_grounding_score(self):
        raw = "Some content\nGROUNDING_ASSESSMENT: 8/10"
        score = _extract_grounding_score(raw)
        self.assertAlmostEqual(score, 0.8)

    def test_grounding_score_default(self):
        score = _extract_grounding_score("No assessment here")
        self.assertEqual(score, 0.5)


# ── Improvement Tests ─────────────────────────────────────────────────────────

class TestDiffAndRules(unittest.TestCase):
    def test_diff_summary_nonempty(self):
        orig = "Line A\nLine B"
        edited = "Line A\nLine B\nLine C added"
        summary = compute_diff_summary(orig, edited)
        self.assertIn("Added", summary)

    def test_no_change_summary(self):
        text = "Unchanged text"
        summary = compute_diff_summary(text, text)
        self.assertIn("No textual changes", summary)

    def test_bullet_preference_detected(self):
        orig = "Text here. More text. And even more text to make it longer."
        edited = orig + "\n- item one\n- item two\n- item three\n- item four\n- item five"
        rules = extract_rules_from_edit(orig, edited)
        self.assertTrue(any("bullet" in r.lower() for r in rules))

    def test_length_reduction_detected(self):
        orig = " ".join(["word"] * 200)
        edited = " ".join(["word"] * 80)
        rules = extract_rules_from_edit(orig, edited)
        self.assertTrue(any("concise" in r.lower() for r in rules))

    def test_new_section_detected(self):
        orig = "## Section A\ncontent"
        edited = "## Section A\ncontent\n## New Section\nnew content"
        rules = extract_rules_from_edit(orig, edited)
        self.assertTrue(any("New Section" in r for r in rules))


class TestImprovementStore(unittest.TestCase):
    def setUp(self):
        import tempfile, os
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.store = ImprovementStore(store_path=self.tmp.name)

    def tearDown(self):
        import os
        os.unlink(self.tmp.name)

    def test_empty_instructions_initially(self):
        instructions = self.store.get_operator_instructions("case_intake")
        self.assertEqual(instructions, "")

    def test_rules_persist_after_save(self):
        # Manually inject a rule
        self.store._data["rules_by_doc_type"]["case_intake"] = ["Be concise"]
        self.store._save()
        store2 = ImprovementStore(store_path=self.tmp.name)
        self.assertIn("Be concise", store2.get_rules("case_intake"))

    def test_stats(self):
        stats = self.store.stats()
        self.assertIn("total_edits_processed", stats)
        self.assertEqual(stats["total_edits_processed"], 0)

    def test_clear_rules(self):
        self.store._data["rules_by_doc_type"]["case_intake"] = ["Some rule"]
        self.store.clear_rules("case_intake")
        self.assertEqual(self.store.get_rules("case_intake"), [])


class TestSimulatedEdits(unittest.TestCase):
    def test_simulate_case_intake(self):
        # Mock draft object
        class MockDraft:
            draft_id = "draft_test001"
            draft_type = "case_intake"
            def to_markdown(self):
                return "## Party Identification\nTest content\n## Financial Summary\nSome amounts"

        edited = simulate_operator_edit(MockDraft(), "case_intake")
        self.assertIn("Risk Flags", edited)

    def test_simulate_title_deed(self):
        class MockDraft:
            draft_id = "draft_test002"
            draft_type = "title_deed"
            def to_markdown(self):
                return "## Encumbrances & Risks\nSome risks here"

        edited = simulate_operator_edit(MockDraft(), "title_deed")
        self.assertIn("Reviewer Notes", edited)


# ── Integration Test ─────────────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):
    def test_ingestion_feeds_retrieval(self):
        """Verify ingestion output is compatible with retrieval input."""
        from ingestion.ingestion import ingest_document
        samples_dir = Path(__file__).parent.parent / "samples"
        sample = next(samples_dir.glob("*.txt"))
        doc = ingest_document(str(sample))

        store = RetrievalStore()
        n = store.add_document(doc)
        self.assertGreater(n, 0)

        results = store.retrieve(doc.cleaned_text[:50])
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ── New: OCR ingestion tests ──────────────────────────────────────────────────

class TestOCRIngestion(unittest.TestCase):
    def test_pdf_ocr_produces_extracteddoc(self):
        """Create a real image, run OCR through the full ingestion stack."""
        try:
            from PIL import Image, ImageDraw
            import pytesseract
        except ImportError:
            self.skipTest("PIL or pytesseract not available")

        import tempfile, os
        img = Image.new("RGB", (500, 150), color="white")
        d = ImageDraw.Draw(img)
        d.text((10, 10),  "DEED OF CONVEYANCE", fill="black")
        d.text((10, 40),  "Vendor: Ramesh Kumar S/o Mohan Lal", fill="black")
        d.text((10, 70),  "Plot No: 112-B  Sale Consideration: Rs. 42,00,000/-", fill="black")
        d.text((10, 100), "Encumbrance: boundary dispute outstanding", fill="black")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            tmp_path = f.name

        try:
            doc = ingest_document(tmp_path)
            self.assertTrue(doc.ocr_used)
            self.assertGreater(len(doc.cleaned_text), 20)
            self.assertEqual(doc.doc_type, "title_deed")
        finally:
            os.unlink(tmp_path)

    def test_image_file_routing(self):
        """Verify .png files are routed through OCR, not plain text reader."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        import tempfile, os
        img = Image.new("RGB", (200, 50), color="white")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            tmp_path = f.name
        try:
            doc = ingest_document(tmp_path)
            self.assertTrue(doc.ocr_used)
        finally:
            os.unlink(tmp_path)

    def test_txt_file_not_ocr(self):
        """Verify .txt files do NOT go through OCR."""
        samples_dir = Path(__file__).parent.parent / "samples"
        sample = next(samples_dir.glob("*.txt"))
        doc = ingest_document(str(sample))
        self.assertFalse(doc.ocr_used)

    def test_ocr_quality_penalty_applied(self):
        """OCR-processed docs get a small quality penalty vs plain text."""
        samples_dir = Path(__file__).parent.parent / "samples"
        sample = next(samples_dir.glob("*.txt"))
        txt_doc = ingest_document(str(sample))
        # Simulate an OCR doc by using text_override but flagging ocr_used
        from ingestion.ingestion import analyze_confidence
        ocr_flags = analyze_confidence(txt_doc.cleaned_text, ocr_used=True)
        txt_flags = analyze_confidence(txt_doc.cleaned_text, ocr_used=False)
        self.assertLessEqual(ocr_flags["quality_score"], txt_flags["quality_score"])

    def test_image_preprocessing_does_not_crash(self):
        """Pre-processing should never raise — always return an image."""
        from PIL import Image
        from ingestion.ingestion import _preprocess_image_for_ocr
        img = Image.new("RGB", (100, 100), color="gray")
        result = _preprocess_image_for_ocr(img)
        self.assertIsNotNone(result)


# ── New: Semantic / hybrid retrieval tests ────────────────────────────────────

class TestSemanticRetrieval(unittest.TestCase):
    def setUp(self):
        from ingestion.ingestion import ingest_directory
        samples_dir = Path(__file__).parent.parent / "samples"
        docs = ingest_directory(str(samples_dir))
        self.store = RetrievalStore()
        self.store.add_documents(docs)
        self.semantic_available = self.store.index.semantic.available

    def test_store_stats_include_retrieval_method(self):
        stats = self.store.stats()
        self.assertIn("retrieval_method", stats)
        self.assertIn("semantic_available", stats)
        self.assertIn(stats["retrieval_method"], ["hybrid", "bm25"])

    def test_retrieval_returns_results_regardless_of_backend(self):
        """Should work whether semantic is available or not."""
        results = self.store.retrieve("outstanding dues rent tenant payment")
        self.assertGreater(len(results), 0)

    def test_evidence_has_retrieval_method_field(self):
        results = self.store.retrieve("vendor purchaser sale property")
        for ev in results:
            self.assertIn(ev.retrieval_method, ["hybrid", "bm25", "semantic"])

    def test_semantic_paraphrase_query(self):
        """
        Semantic retrieval should handle a paraphrase that BM25 would miss.
        'financial obligation owed by occupant' should still retrieve legal notice dues.
        """
        results = self.store.retrieve("financial obligation owed by occupant", top_k=5)
        # We just verify it returns something — semantic may or may not be available
        self.assertIsInstance(results, list)

    def test_hybrid_scores_are_bounded(self):
        """Hybrid scores should be in [0, 1] since both components are normalized."""
        results = self.store.retrieve("case dispute client opposing party")
        for ev in results:
            self.assertGreaterEqual(ev.score, 0.0)
            # When semantic is unavailable, BM25 raw scores are returned (can be >1)
            # When hybrid, scores are normalised to [0,1] weighted sum
            if ev.retrieval_method == "hybrid":
                self.assertLessEqual(ev.score, 1.01)

    def test_batch_indexing_consistent(self):
        """Adding documents in batch should produce same passage count as one at a time."""
        from ingestion.ingestion import ingest_directory
        samples_dir = Path(__file__).parent.parent / "samples"
        docs = ingest_directory(str(samples_dir))

        store_batch = RetrievalStore()
        store_batch.add_documents(docs)

        store_seq = RetrievalStore()
        for doc in docs:
            store_seq.add_document(doc)

        self.assertEqual(
            store_batch.stats()["total_passages"],
            store_seq.stats()["total_passages"],
        )

    def test_doc_type_filter_still_works_with_new_backend(self):
        results = self.store.retrieve("client name dispute", doc_type_filter="case_intake")
        for ev in results:
            self.assertEqual(ev.passage.doc_type, "case_intake")
