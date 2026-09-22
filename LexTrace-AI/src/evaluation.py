"""
evaluation.py — Evaluation framework for LexTrace AI.

Measures four dimensions:

1. Extraction quality
   - document type detection
   - required field coverage
   - document quality threshold

2. Retrieval quality
   - top-1 document-type accuracy
   - precision@3

3. Grounding quality
   - generated claim citation coverage
   - citation validity against the retrieval index
   - section evidence coverage
   - LLM self-reported grounding score (diagnostic only)

4. Improvement quality
   - operator-rule extraction
   - learned instruction generation

Important:
The computed grounding score does NOT trust the LLM's
self-reported GROUNDING_ASSESSMENT.
"""

import sys
import json
import re as regex
import tempfile
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from ingestion.ingestion import ingest_directory
from retrieval.retrieval import RetrievalStore
from generation.generation import GeneratedDraft
from improvement.improvement import (
    ImprovementStore,
    extract_rules_from_edit,
)


# ---------------------------------------------------------------------------
# 1. Extraction evaluation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "title_deed_001.txt": {
        "doc_type": "title_deed",
        "must_have_fields": [
            "sale_consideration",
            "plot_number",
            "encumbrances_present",
        ],
        "expected_doc_type": "title_deed",
        "min_quality_score": 50,
    },

    "legal_notice_002.txt": {
        "doc_type": "legal_notice",
        "must_have_fields": [
            "monthly_rent",
            "total_dues",
            "demands",
        ],
        "expected_doc_type": "legal_notice",
        "min_quality_score": 40,
    },

    "case_intake_003.txt": {
        "doc_type": "case_intake",
        "must_have_fields": [
            "client_name",
            "opposing_party",
            "nature_of_dispute",
            "documents_present",
        ],
        "expected_doc_type": "case_intake",
        "min_quality_score": 70,
    },
}


def evaluate_extraction(docs: list) -> dict:
    """Evaluate structured extraction against known sample ground truth."""

    scores = {}

    for doc in docs:
        filename = Path(doc.source_path).name
        ground_truth = GROUND_TRUTH.get(filename)

        if not ground_truth:
            continue

        result = {
            "doc_id": doc.doc_id,
            "type_correct": (
                doc.doc_type
                == ground_truth.get("expected_doc_type")
            ),
            "quality_score": doc.confidence_flags.get(
                "quality_score",
                0,
            ),
            "quality_passes": (
                doc.confidence_flags.get(
                    "quality_score",
                    0,
                )
                >= ground_truth.get(
                    "min_quality_score",
                    0,
                )
            ),
            "fields_found": [],
            "fields_missing": [],
        }

        for field_name in ground_truth.get(
            "must_have_fields",
            [],
        ):
            value = doc.structured_fields.get(field_name)

            if value:
                result["fields_found"].append(field_name)
            else:
                result["fields_missing"].append(field_name)

        total_fields = len(
            ground_truth.get(
                "must_have_fields",
                [],
            )
        )

        result["field_coverage"] = (
            len(result["fields_found"]) / total_fields
            if total_fields
            else 1.0
        )

        scores[filename] = result

    if scores:
        avg_coverage = sum(
            result["field_coverage"]
            for result in scores.values()
        ) / len(scores)

        type_accuracy = sum(
            1
            for result in scores.values()
            if result["type_correct"]
        ) / len(scores)

        quality_pass_rate = sum(
            1
            for result in scores.values()
            if result["quality_passes"]
        ) / len(scores)

    else:
        avg_coverage = 0.0
        type_accuracy = 0.0
        quality_pass_rate = 0.0

    overall = (
        avg_coverage
        + type_accuracy
        + quality_pass_rate
    ) / 3

    return {
        "per_doc": scores,
        "aggregate": {
            "avg_field_coverage": round(
                avg_coverage,
                3,
            ),
            "type_detection_accuracy": round(
                type_accuracy,
                3,
            ),
            "quality_score_pass_rate": round(
                quality_pass_rate,
                3,
            ),
            "overall_score": round(
                overall,
                3,
            ),
        },
    }


# ---------------------------------------------------------------------------
# 2. Retrieval evaluation
# ---------------------------------------------------------------------------

RETRIEVAL_QUERIES = [
    {
        "query": (
            "outstanding dues rent payment "
            "tenant violation"
        ),
        "expected_doc_type": "legal_notice",
        "description": "Legal notice dues query",
    },

    {
        "query": (
            "vendor purchaser sale consideration plot"
        ),
        "expected_doc_type": "title_deed",
        "description": "Title deed parties query",
    },

    {
        "query": (
            "client name opposing party "
            "nature of dispute RERA"
        ),
        "expected_doc_type": "case_intake",
        "description": "Case intake facts query",
    },
]


def evaluate_retrieval(store: RetrievalStore) -> dict:
    """Evaluate retrieval against known document-type expectations."""

    results = []

    for query_spec in RETRIEVAL_QUERIES:

        retrieved = store.retrieve(
            query_spec["query"],
            top_k=5,
        )

        if not retrieved:
            results.append(
                {
                    "query": query_spec["description"],
                    "top_result_correct_type": False,
                    "precision_at_3": 0.0,
                    "returned_count": 0,
                }
            )
            continue

        top_3 = retrieved[:3]

        correct_in_top3 = sum(
            1
            for evidence in top_3
            if (
                evidence.passage.doc_type
                == query_spec["expected_doc_type"]
            )
        )

        results.append(
            {
                "query": query_spec["description"],
                "top_result_correct_type": (
                    retrieved[0].passage.doc_type
                    == query_spec["expected_doc_type"]
                ),
                "precision_at_3": (
                    correct_in_top3 / len(top_3)
                    if top_3
                    else 0.0
                ),
                "returned_count": len(retrieved),
                "top_score": round(
                    retrieved[0].score,
                    3,
                ),
            }
        )

    if results:
        avg_precision_at_3 = sum(
            result["precision_at_3"]
            for result in results
        ) / len(results)

        top1_accuracy = sum(
            1
            for result in results
            if result["top_result_correct_type"]
        ) / len(results)

    else:
        avg_precision_at_3 = 0.0
        top1_accuracy = 0.0

    overall = (
        avg_precision_at_3
        + top1_accuracy
    ) / 2

    return {
        "per_query": results,
        "aggregate": {
            "avg_precision_at_3": round(
                avg_precision_at_3,
                3,
            ),
            "top1_accuracy": round(
                top1_accuracy,
                3,
            ),
            "overall_score": round(
                overall,
                3,
            ),
        },
    }


# ---------------------------------------------------------------------------
# 3. Grounding evaluation
# ---------------------------------------------------------------------------

def extract_citation_ids(text: str) -> list[str]:
    """
    Extract LexTrace passage IDs from generated text.

    Handles:

        [doc_xxx_page1_chunk0]

    and:

        [doc_xxx_page1_chunk0, Page 1]

    and:

        [doc_xxx_page1_chunk0, doc_xxx_page1_chunk1, Page 1]
    """

    if not text:
        return []

    pattern = (
        r"\b"
        r"(doc_[a-zA-Z0-9]+"
        r"(?:_page\d+)?"
        r"_chunk\d+)"
        r"\b"
    )

    return regex.findall(
        pattern,
        text,
    )


def extract_claim_units(text: str) -> list[str]:
    """
    Extract claim-bearing lines from generated Markdown.

    Headings and metadata are ignored.
    Evidence blocks are ignored because they are supporting
    material rather than generated claims.
    """

    if not text:
        return []

    claims = []

    blocks = regex.split(
        r"\n\s*\n",
        text,
    )

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        lines = block.splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # Ignore Markdown headings.
            if line.startswith("#"):
                continue

            # Ignore evidence label.
            if line.startswith("**Evidence:**"):
                continue

            # Ignore blockquotes used for metadata.
            if line.startswith(">"):
                continue

            # Ignore model's grounding assessment.
            if line.upper().startswith(
                "GROUNDING_ASSESSMENT:"
            ):
                continue

            # Bullet/list claims.
            if line.startswith("-") or line.startswith("*"):
                claims.append(line)
                continue

            # Numbered claims.
            if regex.match(
                r"^\d+\.",
                line,
            ):
                claims.append(line)
                continue

            # Normal prose.
            claims.append(line)

    return claims


def load_generated_drafts(
    output_dir: str = "outputs",
) -> list[GeneratedDraft]:
    """
    Load actual GeneratedDraft objects from pipeline JSON outputs.

    The main pipeline already saves:
        outputs/<draft_id>_draft.json

    Therefore evaluation does NOT need to call Gemini again.
    """

    output_path = Path(output_dir)

    if not output_path.exists():
        return []

    draft_files = sorted(
        output_path.glob("*_draft.json")
    )

    drafts = []

    for draft_file in draft_files:

        try:
            data = json.loads(
                draft_file.read_text(
                    encoding="utf-8"
                )
            )

            sections = []

            for section_data in data.get(
                "sections",
                [],
            ):
                from generation.generation import DraftSection

                sections.append(
                    DraftSection(
                        section_name=section_data.get(
                            "section_name",
                            "",
                        ),
                        content=section_data.get(
                            "content",
                            "",
                        ),
                        supporting_passages=section_data.get(
                            "supporting_passages",
                            [],
                        ),
                        grounding_score=float(
                            section_data.get(
                                "grounding_score",
                                0.0,
                            )
                        ),
                    )
                )

            draft = GeneratedDraft(
                draft_id=data.get(
                    "draft_id",
                    draft_file.stem.replace(
                        "_draft",
                        "",
                    ),
                ),
                doc_ids=data.get(
                    "doc_ids",
                    [],
                ),
                draft_type=data.get(
                    "draft_type",
                    "unknown",
                ),
                title=data.get(
                    "title",
                    draft_file.stem,
                ),
                sections=sections,
                overall_grounding_score=float(
                    data.get(
                        "overall_grounding_score",
                        0.0,
                    )
                ),
                generated_at=data.get(
                    "generated_at",
                    "",
                ),
                model_used=data.get(
                    "model_used",
                    "unknown",
                ),
                operator_instructions=data.get(
                    "operator_instructions",
                    "",
                ),
                raw_llm_response=data.get(
                    "raw_llm_response",
                    "",
                ),
            )

            drafts.append(draft)

        except Exception as exc:
            print(
                f"  Warning: could not load "
                f"{draft_file.name}: {exc}"
            )

    return drafts


def evaluate_grounding(
    drafts: list,
    retrieval_store: RetrievalStore,
) -> dict:
    """
    Evaluate grounding using observable citation behavior.

    The computed grounding score is based on:

        60% citation coverage
        40% citation validity

    LLM self-reported grounding is retained only as a diagnostic.

    Citation coverage:
        Fraction of generated claim units containing
        at least one passage citation.

    Citation validity:
        Fraction of cited passage IDs that actually exist
        in the current retrieval index.
    """

    # Build authoritative set of indexed passage IDs.
    indexed_passage_ids = {
        passage.passage_id
        for passage in retrieval_store._all_passages
    }

    results = []

    for draft in drafts:

        all_claims = []
        all_citations = []

        section_results = []

        for section in draft.sections:

            claims = extract_claim_units(
                section.content
            )

            citations = extract_citation_ids(
                section.content
            )

            all_claims.extend(claims)
            all_citations.extend(citations)

            cited_claims = sum(
                1
                for claim in claims
                if extract_citation_ids(claim)
            )

            section_results.append(
                {
                    "section_name": section.section_name,
                    "claim_count": len(claims),
                    "claims_with_citations": cited_claims,
                    "citation_count": len(citations),
                    "supporting_passages": len(
                        section.supporting_passages or []
                    ),
                    "citation_coverage": (
                        cited_claims / len(claims)
                        if claims
                        else 0.0
                    ),
                }
            )

        claim_count = len(all_claims)

        claims_with_citations = sum(
            1
            for claim in all_claims
            if extract_citation_ids(claim)
        )

        citation_count = len(all_citations)

        valid_citations = [
            citation
            for citation in all_citations
            if citation in indexed_passage_ids
        ]

        citation_coverage = (
            claims_with_citations / claim_count
            if claim_count
            else 0.0
        )

        citation_validity = (
            len(valid_citations) / citation_count
            if citation_count
            else 0.0
        )

        sections_with_evidence = sum(
            1
            for section in draft.sections
            if section.supporting_passages
        )

        total_sections = len(
            draft.sections
        )

        section_evidence_coverage = (
            sections_with_evidence / total_sections
            if total_sections
            else 0.0
        )

        computed_grounding = (
            citation_coverage * 0.60
            + citation_validity * 0.40
        )

        results.append(
            {
                "draft_id": draft.draft_id,
                "doc_type": draft.draft_type,

                # Diagnostic only.
                "llm_grounding_score": round(
                    draft.overall_grounding_score,
                    3,
                ),

                "claim_count": claim_count,
                "claims_with_citations": (
                    claims_with_citations
                ),
                "citation_count": citation_count,
                "valid_citation_count": len(
                    valid_citations
                ),

                "citation_coverage": round(
                    citation_coverage,
                    3,
                ),

                "citation_validity": round(
                    citation_validity,
                    3,
                ),

                "sections_with_evidence": (
                    sections_with_evidence
                ),
                "total_sections": total_sections,

                "section_evidence_coverage": round(
                    section_evidence_coverage,
                    3,
                ),

                "computed_grounding_score": round(
                    computed_grounding,
                    3,
                ),

                "sections": section_results,
            }
        )

    if results:

        avg_llm_score = sum(
            result["llm_grounding_score"]
            for result in results
        ) / len(results)

        avg_citation_coverage = sum(
            result["citation_coverage"]
            for result in results
        ) / len(results)

        avg_citation_validity = sum(
            result["citation_validity"]
            for result in results
        ) / len(results)

        avg_section_coverage = sum(
            result["section_evidence_coverage"]
            for result in results
        ) / len(results)

        avg_computed_grounding = sum(
            result["computed_grounding_score"]
            for result in results
        ) / len(results)

    else:
        avg_llm_score = 0.0
        avg_citation_coverage = 0.0
        avg_citation_validity = 0.0
        avg_section_coverage = 0.0
        avg_computed_grounding = 0.0

    return {
        "per_draft": results,

        "aggregate": {
            # Diagnostic only.
            "avg_llm_grounding_score": round(
                avg_llm_score,
                3,
            ),

            "avg_citation_coverage": round(
                avg_citation_coverage,
                3,
            ),

            "avg_citation_validity": round(
                avg_citation_validity,
                3,
            ),

            "avg_section_evidence_coverage": round(
                avg_section_coverage,
                3,
            ),

            "computed_grounding_score": round(
                avg_computed_grounding,
                3,
            ),

            # Keep this name for compatibility with
            # the report summary.
            "overall_score": round(
                avg_computed_grounding,
                3,
            ),
        },
    }


# ---------------------------------------------------------------------------
# 4. Improvement evaluation
# ---------------------------------------------------------------------------

def evaluate_improvement() -> dict:
    """
    Test whether the improvement loop can:

    - detect formatting preferences
    - detect conciseness preferences
    - detect new sections
    - create reusable operator instructions
    """

    results = {}

    test_cases = [
        {
            "name": "bullet_preference",
            "orig": (
                "Some text here. "
                "More sentences follow."
            ),
            "edited": (
                "Some text here. "
                "More sentences follow.\n"
                "- item 1\n"
                "- item 2\n"
                "- item 3\n"
                "- item 4\n"
                "- item 5"
            ),
            "expected_keyword": "bullet",
        },

        {
            "name": "conciseness",
            "orig": " ".join(
                ["word"] * 200
            ),
            "edited": " ".join(
                ["word"] * 80
            ),
            "expected_keyword": "concise",
        },

        {
            "name": "new_section",
            "orig": (
                "## Section A\n"
                "content"
            ),
            "edited": (
                "## Section A\n"
                "content\n"
                "## Risk Analysis\n"
                "new important section"
            ),
            "expected_keyword": "Risk Analysis",
        },

        {
            "name": "citation_style",
            "orig": (
                "Some statement."
            ),
            "edited": (
                "Some statement "
                "[doc_test_page1_chunk0].\n"
                "Another statement "
                "[doc_test_page1_chunk1].\n"
                "Third statement "
                "[doc_test_page1_chunk2]."
            ),
            "expected_keyword": "citation",
        },
    ]

    passed = 0

    for case in test_cases:

        rules = extract_rules_from_edit(
            case["orig"],
            case["edited"],
        )

        found = any(
            case["expected_keyword"].lower()
            in rule.lower()
            for rule in rules
        )

        results[case["name"]] = {
            "rules_extracted": rules,
            "expected_found": found,
        }

        if found:
            passed += 1

    # Test persistent operator instructions.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

        improvement_store = ImprovementStore(
            store_path=temp_path
        )

        improvement_store._data[
            "rules_by_doc_type"
        ]["case_intake"] = [
            "Be concise",
            "Use bullet points",
        ]

        improvement_store._save()

        instructions = (
            improvement_store
            .get_operator_instructions(
                "case_intake"
            )
        )

        has_instructions = bool(
            instructions
        )

    finally:

        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()

    rule_accuracy = (
        passed / len(test_cases)
        if test_cases
        else 0.0
    )

    overall = (
        rule_accuracy
        + int(has_instructions)
    ) / 2

    return {
        "rule_extraction_tests": results,

        "rule_extraction_pass_rate": round(
            rule_accuracy,
            3,
        ),

        "instructions_generated": has_instructions,

        "aggregate": {
            "rule_extraction_accuracy": round(
                rule_accuracy,
                3,
            ),
            "overall_score": round(
                overall,
                3,
            ),
        },
    }


# ---------------------------------------------------------------------------
# 5. Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    samples_dir: str = "samples",
    output_dir: str = "outputs",
    output_path: str = (
        "outputs/evaluation_report.json"
    ),
) -> dict:

    print("Running evaluation...\n")

    # -------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------

    docs = ingest_directory(
        samples_dir
    )

    # -------------------------------------------------------
    # Retrieval index
    # -------------------------------------------------------

    retrieval_store = RetrievalStore()

    retrieval_store.add_documents(
        docs
    )

    # -------------------------------------------------------
    # Load ACTUAL generated drafts
    # -------------------------------------------------------

    drafts = load_generated_drafts(
        output_dir
    )

    if not drafts:

        print(
            "ERROR: No generated draft JSON files "
            "were found."
        )

        print(
            f"Expected files such as: "
            f"{output_dir}\\*_draft.json"
        )

        print(
            "\nRun the pipeline first:"
        )

        print(
            "python src\\pipeline.py "
            "--input samples\\"
        )

        return {
            "error": (
                "No generated draft JSON files found."
            )
        }

    print(
        f"Loaded {len(drafts)} generated draft(s)."
    )

    # -------------------------------------------------------
    # Run evaluations
    # -------------------------------------------------------

    extraction_eval = evaluate_extraction(
        docs
    )

    retrieval_eval = evaluate_retrieval(
        retrieval_store
    )

    grounding_eval = evaluate_grounding(
        drafts,
        retrieval_store,
    )

    improvement_eval = evaluate_improvement()

    # -------------------------------------------------------
    # Overall score
    # -------------------------------------------------------

    extraction_score = (
        extraction_eval[
            "aggregate"
        ]["overall_score"]
    )

    retrieval_score = (
        retrieval_eval[
            "aggregate"
        ]["overall_score"]
    )

    grounding_score = (
        grounding_eval[
            "aggregate"
        ]["overall_score"]
    )

    improvement_score = (
        improvement_eval[
            "aggregate"
        ]["overall_score"]
    )

    # These are development-time weights.
    # They should NOT be presented as a benchmark accuracy.
    weighted_score = (
        extraction_score * 25
        + retrieval_score * 25
        + grounding_score * 10
        + improvement_score * 25
    ) / 85

    # -------------------------------------------------------
    # Build report
    # -------------------------------------------------------

    report = {
        "evaluation_version": "1.0",

        "notes": [
            (
                "Grounding score is computed from "
                "citation coverage and citation validity."
            ),
            (
                "LLM self-reported grounding is retained "
                "only as a diagnostic."
            ),
            (
                "The weighted overall score is a "
                "development rubric, not an accuracy claim."
            ),
        ],

        "extraction": extraction_eval,

        "retrieval": retrieval_eval,

        "grounding": grounding_eval,

        "improvement": improvement_eval,

        "summary": {
            "extraction_score": round(
                extraction_score,
                3,
            ),

            "retrieval_score": round(
                retrieval_score,
                3,
            ),

            "grounding_score": round(
                grounding_score,
                3,
            ),

            "improvement_score": round(
                improvement_score,
                3,
            ),

            "weighted_overall": round(
                weighted_score,
                3,
            ),
        },
    }

    # -------------------------------------------------------
    # Save report
    # -------------------------------------------------------

    report_path = Path(
        output_path
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------
    # Console output
    # -------------------------------------------------------

    print(
        "\n=== EVALUATION RESULTS ==="
    )

    print(
        f"  Extraction:              "
        f"{extraction_score:.1%}"
    )

    print(
        f"  Retrieval:               "
        f"{retrieval_score:.1%}"
    )

    print(
        f"  Grounding:               "
        f"{grounding_score:.1%}"
    )

    print(
        f"  Citation coverage:       "
        f"{grounding_eval['aggregate']['avg_citation_coverage']:.1%}"
    )

    print(
        f"  Citation validity:       "
        f"{grounding_eval['aggregate']['avg_citation_validity']:.1%}"
    )

    print(
        f"  Section evidence:        "
        f"{grounding_eval['aggregate']['avg_section_evidence_coverage']:.1%}"
    )

    print(
        f"  LLM grounding "
        f"(diagnostic):            "
        f"{grounding_eval['aggregate']['avg_llm_grounding_score']:.1%}"
    )

    print(
        f"  Improvement:             "
        f"{improvement_score:.1%}"
    )

    print(
        "  ─────────────────────────────"
    )

    print(
        f"  Weighted development:    "
        f"{weighted_score:.1%}"
    )

    print(
        f"\n  Report saved: "
        f"{report_path}"
    )

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()