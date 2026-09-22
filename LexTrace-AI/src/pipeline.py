"""
pipeline.py — Top-level orchestrator.

Wires together ingestion → retrieval → generation → improvement.
Can be run as a script or imported as a module.

Usage:
    python pipeline.py --input samples/title_deed_001.txt
    python pipeline.py --input samples/ --simulate-edit
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.ingestion import ingest_document, ingest_directory
from retrieval.retrieval import RetrievalStore
from generation.generation import generate_draft
from improvement.improvement import ImprovementStore, simulate_operator_edit


def run_pipeline(
    input_path: str,
    simulate_edit: bool = False,
    output_dir: str = "outputs",
    store_path: str = "improvement_store.json",
    verbose: bool = True,
) -> dict:
    """
    Full pipeline run.
    Returns a dict with all outputs for inspection/testing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # ── 1. INGESTION ──────────────────────────────────────────────────────
    if verbose:
        print("\n[1/4] Ingesting documents...")

    p = Path(input_path)
    if p.is_dir():
        docs = ingest_directory(input_path)
    else:
        docs = [ingest_document(input_path)]

    if verbose:
        for doc in docs:
            print(f"  OK {doc.doc_id} | type={doc.doc_type} | quality={doc.confidence_flags['quality_score']} | warnings={len(doc.warnings)}")

    # Save extracted output
    for doc in docs:
        out_file = output_path / f"{doc.doc_id}_extracted.json"
        out_file.write_text(json.dumps(doc.to_dict(), indent=2))
        if verbose:
            print(f"  -> Saved extraction: {out_file}")

    # ── 2. RETRIEVAL ──────────────────────────────────────────────────────
    if verbose:
        print("\n[2/4] Building retrieval index...")

    store = RetrievalStore()
    index_summary = store.add_documents(docs)

    if verbose:
        stats = store.stats()
        print(f"  OK Indexed {stats['total_passages']} passages across {stats['total_documents']} doc(s)")

    # ── 3. GENERATION ─────────────────────────────────────────────────────
    if verbose:
        print("\n[3/4] Generating grounded drafts...")

    improvement_store = ImprovementStore(store_path=store_path)
    drafts = []

    for doc in docs:
        operator_instructions = improvement_store.get_operator_instructions(doc.doc_type)
        if verbose and operator_instructions:
            print(f"  ℹ Applying {len(improvement_store.get_rules(doc.doc_type))} learned rule(s) for {doc.doc_type}")

        draft = generate_draft(
            extracted_doc=doc,
            retrieval_store=store,
            operator_instructions=operator_instructions,
        )
        drafts.append(draft)

        # Save draft
        draft_json = output_path / f"{draft.draft_id}_draft.json"
        draft_json.write_text(json.dumps(draft.to_dict(), indent=2))
        draft_md = output_path / f"{draft.draft_id}_draft.md"
        draft_md.write_text(draft.to_markdown())

        if verbose:
            print(f"  OK Draft {draft.draft_id} | grounding={draft.overall_grounding_score:.0%} | sections={len(draft.sections)}")
            print(f"  -> Saved: {draft_md}")

    # ── 4. IMPROVEMENT ────────────────────────────────────────────────────
    if simulate_edit:
        if verbose:
            print("\n[4/4] Simulating operator edits and extracting improvement rules...")

        edit_records = []
        for doc, draft in zip(docs, drafts):
            edited_md = simulate_operator_edit(draft, doc.doc_type)
            edit_record = improvement_store.record_edit(draft, edited_md)
            edit_records.append(edit_record)

            if verbose:
                print(f"  OK Edit {edit_record.edit_id}: {edit_record.diff_summary[:100]}")
                if edit_record.extracted_rules:
                    for rule in edit_record.extracted_rules:
                        print(f"    -> Rule learned: {rule}")
                else:
                    print(f"    -> No new rules extracted (edit was minor)")

            # Save edit record
            edit_file = output_path / f"{edit_record.edit_id}_edit.json"
            edit_file.write_text(json.dumps(edit_record.to_dict(), indent=2))

        if verbose:
            print(f"\n  📊 Improvement store stats: {improvement_store.stats()}")
    else:
        edit_records = []
        if verbose:
            print("\n[4/4] Skipping operator edit simulation (pass --simulate-edit to enable)")

    if verbose:
        print(f"\n✅ Pipeline complete. Outputs in: {output_path}/")

    return {
        "docs": docs,
        "drafts": drafts,
        "edit_records": edit_records,
        "improvement_stats": improvement_store.stats(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ambitio Legal Document AI Pipeline")
    parser.add_argument("--input", default="samples/", help="Input file or directory")
    parser.add_argument("--simulate-edit", action="store_true", help="Simulate operator edits")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--store", default="improvement_store.json", help="Improvement store path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        simulate_edit=args.simulate_edit,
        output_dir=args.output_dir,
        store_path=args.store,
        verbose=not args.quiet,
    )
