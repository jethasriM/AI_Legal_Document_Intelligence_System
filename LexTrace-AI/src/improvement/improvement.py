"""
improvement.py — Learning from operator edits.

When an operator reviews and edits a generated draft, this module:
  1. Captures the diff between original and edited draft
  2. Analyzes what changed (additions, deletions, rewrites, tone shifts)
  3. Extracts reusable patterns as operator style instructions
  4. Persists those instructions so future drafts start better

The improvement loop is implemented as a growing set of extracted "style rules"
that are injected into the generation prompt as operator instructions.
This is deliberately simple and inspectable — operators can see and override rules.
"""

import re
import json
import difflib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Edit capture
# ---------------------------------------------------------------------------

@dataclass
class EditRecord:
    edit_id: str
    draft_id: str
    doc_type: str
    original_draft_md: str
    edited_draft_md: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    extracted_rules: list[str] = field(default_factory=list)
    diff_summary: str = ""

    def to_dict(self):
        return asdict(self)


def compute_diff_summary(original: str, edited: str) -> str:
    """
    Produce a human-readable summary of what changed.
    Used for analysis and debugging.
    """
    orig_lines = original.splitlines()
    edit_lines = edited.splitlines()
    diff = list(difflib.unified_diff(orig_lines, edit_lines, lineterm="", n=0))

    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]

    summary_parts = []
    if added:
        summary_parts.append(f"Added {len(added)} line(s): e.g. \"{added[0][:80]}\"")
    if removed:
        summary_parts.append(f"Removed {len(removed)} line(s): e.g. \"{removed[0][:80]}\"")
    if not added and not removed:
        summary_parts.append("No textual changes detected")

    return " | ".join(summary_parts)


# ---------------------------------------------------------------------------
# Pattern extraction from edits
# ---------------------------------------------------------------------------

# Each detector returns a rule string if it fires, else None
def _detect_section_reordering(orig: str, edited: str) -> Optional[str]:
    orig_headers = re.findall(r'^## (.+)$', orig, re.MULTILINE)
    edit_headers = re.findall(r'^## (.+)$', edited, re.MULTILINE)
    if orig_headers != edit_headers and set(orig_headers) == set(edit_headers):
        return f"Preferred section order: {' → '.join(edit_headers)}"
    return None


def _detect_added_legal_language(orig: str, edited: str) -> Optional[str]:
    """Detect if operator consistently adds formal legal phrasing."""
    legal_terms = [
        "without prejudice", "notwithstanding", "pursuant to",
        "inter alia", "prima facie", "bona fide", "ex parte",
        "hereby", "hereinafter", "aforementioned", "as per",
    ]
    orig_lower = orig.lower()
    edit_lower = edited.lower()
    added_terms = [t for t in legal_terms if t not in orig_lower and t in edit_lower]
    if len(added_terms) >= 2:
        return f"Use formal legal language including terms like: {', '.join(added_terms[:3])}"
    return None


def _detect_bullet_preference(orig: str, edited: str) -> Optional[str]:
    orig_bullets = orig.count("\n-") + orig.count("\n*")
    edit_bullets = edited.count("\n-") + edited.count("\n*")
    if edit_bullets > orig_bullets + 3:
        return "Prefer bullet-point lists over prose paragraphs for itemized information"
    if edit_bullets < orig_bullets - 3:
        return "Prefer prose paragraphs over bullet lists — write in flowing sentences"
    return None


def _detect_length_preference(orig: str, edited: str) -> Optional[str]:
    orig_len = len(orig.split())
    edit_len = len(edited.split())
    ratio = edit_len / max(orig_len, 1)
    if ratio > 1.4:
        return "Be more thorough and detailed — expand explanations and include more context"
    if ratio < 0.65:
        return "Be concise — trim unnecessary preamble, keep each section brief and punchy"
    return None


def _detect_missing_section_added(orig: str, edited: str) -> Optional[str]:
    orig_headers = set(re.findall(r'^## (.+)$', orig, re.MULTILINE))
    edit_headers = set(re.findall(r'^## (.+)$', edited, re.MULTILINE))
    new_sections = edit_headers - orig_headers
    if new_sections:
        return f"Always include these additional sections: {', '.join(new_sections)}"
    return None


def _detect_risk_flag_additions(orig: str, edited: str) -> Optional[str]:
    risk_terms = ["risk", "warning", "caution", "flag", "concern", "issue", "problem", "missing", "not found"]
    orig_risk = sum(orig.lower().count(t) for t in risk_terms)
    edit_risk = sum(edited.lower().count(t) for t in risk_terms)
    if edit_risk > orig_risk + 3:
        return "Explicitly flag risks, missing documents, and ambiguities — don't minimize concerns"
    return None


def _detect_citation_style(orig: str, edited: str) -> Optional[str]:
    orig_inline_cites = len(re.findall(r'\[doc_\w+', orig))
    edit_inline_cites = len(re.findall(r'\[doc_\w+', edited))
    if edit_inline_cites > orig_inline_cites + 2:
        return "Include explicit source citations [passage_id] throughout the draft"
    if edit_inline_cites < orig_inline_cites - 2:
        return "Minimize inline citations — use evidence footnotes only at section end"
    return None


PATTERN_DETECTORS = [
    _detect_section_reordering,
    _detect_added_legal_language,
    _detect_bullet_preference,
    _detect_length_preference,
    _detect_missing_section_added,
    _detect_risk_flag_additions,
    _detect_citation_style,
]


def extract_rules_from_edit(original: str, edited: str) -> list[str]:
    """Run all detectors and return list of extracted style rules."""
    rules = []
    for detector in PATTERN_DETECTORS:
        result = detector(original, edited)
        if result:
            rules.append(result)
    return rules


# ---------------------------------------------------------------------------
# Persistent rule store
# ---------------------------------------------------------------------------

class ImprovementStore:
    """
    Manages the growing set of operator style rules per document type.
    Rules are persisted to a JSON file so they survive restarts.
    New drafts load the current ruleset and inject it into the prompt.
    """

    def __init__(self, store_path: str = "improvement_store.json"):
        self.store_path = Path(store_path)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.store_path.exists():
            try:
                return json.loads(self.store_path.read_text())
            except Exception:
                pass
        return {
            "rules_by_doc_type": {},
            "edit_history": [],
            "total_edits_processed": 0,
        }

    def _save(self):
        self.store_path.write_text(json.dumps(self._data, indent=2))

    def record_edit(self, draft, edited_md: str) -> EditRecord:
        """
        Process an operator edit.
        - Compute diff + summary
        - Extract rules
        - Update rule store
        - Return the EditRecord for logging
        """
        import hashlib

        original_md = draft.to_markdown() if hasattr(draft, "to_markdown") else str(draft)
        diff_summary = compute_diff_summary(original_md, edited_md)
        new_rules = extract_rules_from_edit(original_md, edited_md)

        edit_id = "edit_" + hashlib.md5(
            (draft.draft_id + datetime.utcnow().isoformat()).encode()
        ).hexdigest()[:8]

        edit = EditRecord(
            edit_id=edit_id,
            draft_id=draft.draft_id,
            doc_type=draft.draft_type,
            original_draft_md=original_md,
            edited_draft_md=edited_md,
            extracted_rules=new_rules,
            diff_summary=diff_summary,
        )

        # Merge rules: new rules for this doc_type get added; duplicates deduplicated
        doc_type = draft.draft_type
        existing_rules = self._data["rules_by_doc_type"].get(doc_type, [])
        merged = list(dict.fromkeys(existing_rules + new_rules))  # preserves order, deduplicates
        self._data["rules_by_doc_type"][doc_type] = merged

        # Append to history (keep last 50)
        self._data["edit_history"].append(edit.to_dict())
        self._data["edit_history"] = self._data["edit_history"][-50:]
        self._data["total_edits_processed"] += 1

        self._save()
        return edit

    def get_operator_instructions(self, doc_type: str) -> str:
        """
        Build the operator instructions string to inject into the next draft prompt.
        """
        rules = self._data["rules_by_doc_type"].get(doc_type, [])
        if not rules:
            return ""
        lines = ["Based on past operator edits, follow these style preferences:"]
        for i, rule in enumerate(rules, 1):
            lines.append(f"{i}. {rule}")
        return "\n".join(lines)

    def get_rules(self, doc_type: str) -> list[str]:
        return self._data["rules_by_doc_type"].get(doc_type, [])

    def stats(self) -> dict:
        return {
            "total_edits_processed": self._data["total_edits_processed"],
            "doc_types_with_rules": list(self._data["rules_by_doc_type"].keys()),
            "rules_by_doc_type": {
                k: len(v) for k, v in self._data["rules_by_doc_type"].items()
            },
        }

    def clear_rules(self, doc_type: Optional[str] = None):
        """Reset rules — useful for testing."""
        if doc_type:
            self._data["rules_by_doc_type"].pop(doc_type, None)
        else:
            self._data["rules_by_doc_type"] = {}
        self._save()


# ---------------------------------------------------------------------------
# Simulated operator edits (for demo/testing purposes)
# ---------------------------------------------------------------------------

SIMULATED_EDITS = {
    "case_intake": lambda md: (
        md
        .replace("## Party Identification", "## Parties Involved")
        .replace("[Section not generated]", "NOT FOUND IN DOCUMENTS — requires follow-up")
        + "\n\n## Risk Flags\n- RERA registration to be verified urgently\n- Missing payment receipts (3 of 9)\n- Force majeure claim likely weak — flag for counsel"
    ),
    "title_deed": lambda md: (
        md
        .replace("## Encumbrances & Risks", "## ⚠️ Encumbrances, Risks & Warnings")
        + "\n\n## Reviewer Notes\n- Municipal clearance missing — DO NOT proceed without this\n- Boundary dispute must be resolved before registration"
    ),
    "legal_notice": lambda md: (
        md
        .replace("## Demands Made", "## Demands & Deadlines")
        .replace("notwithstanding", "notwithstanding")  # noop, but realistic
        + "\n\n## Recommended Response Timeline\n- Day 1-3: Acknowledge receipt\n- Day 7: Respond on dues calculation\n- Day 15: Final compliance or counter-notice"
    ),
}


def simulate_operator_edit(draft, doc_type: str) -> str:
    """
    Apply a canned edit to a draft markdown — simulates an operator reviewing
    and improving the output. Returns the edited markdown string.
    """
    original_md = draft.to_markdown() if hasattr(draft, "to_markdown") else str(draft)
    edit_fn = SIMULATED_EDITS.get(doc_type, lambda md: md + "\n\n## Additional Notes\nReviewed by operator.")
    return edit_fn(original_md)
