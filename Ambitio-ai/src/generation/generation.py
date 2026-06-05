"""
generation.py — Grounded draft generation.

Uses Google Gemini as the LLM backend (gemini-2.0-flash).

Setup:
  pip install google-generativeai
  export GEMINI_API_KEY=AIza...

Each section of the draft is explicitly anchored to passage citations.
Output format: case fact summary (the chosen draft type).
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Draft data structures
# ---------------------------------------------------------------------------

@dataclass
class DraftSection:
    section_name: str
    content: str
    supporting_passages: list[dict]   # [{passage_id, doc_id, excerpt, score}]
    grounding_score: float            # 0-1, fraction of content backed by evidence


@dataclass
class GeneratedDraft:
    draft_id: str
    doc_ids: list[str]
    draft_type: str
    title: str
    sections: list[DraftSection]
    overall_grounding_score: float
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_used: str = "gemini-2.0-flash"
    operator_instructions: str = ""   # injected style/format preferences from improvement loop
    raw_llm_response: str = ""

    def to_dict(self):
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}\n"]
        lines.append(f"*Generated: {self.generated_at} | Grounding: {self.overall_grounding_score:.0%}*\n")
        if self.operator_instructions:
            lines.append(f"> **Style applied:** {self.operator_instructions}\n")
        for sec in self.sections:
            lines.append(f"\n## {sec.section_name}\n")
            lines.append(sec.content)
            if sec.supporting_passages:
                lines.append("\n\n**Evidence:**")
                for p in sec.supporting_passages:
                    lines.append(f"\n- `{p['passage_id']}` (score {p['score']:.2f}): *\"{p['excerpt']}\"*")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

DRAFT_SECTION_QUERIES = {
    "case_intake": {
        "Party Identification": "client name opposing party contact address",
        "Nature of Dispute": "nature dispute facts alleged complaint",
        "Financial Summary": "amounts money paid outstanding dues consideration",
        "Document Status": "documents attached missing required available",
        "Preliminary Assessment": "legal assessment next steps limitation rera",
    },
    "title_deed": {
        "Parties": "vendor purchaser seller buyer names",
        "Property Description": "plot survey area locality address property",
        "Transaction Details": "sale consideration amount paid advance balance",
        "Encumbrances & Risks": "encumbrance dispute mortgage tax arrears pending",
        "Document Completeness": "attached missing certificate registration documents",
    },
    "legal_notice": {
        "Parties": "client recipient tenant landlord notice",
        "Tenancy Background": "lease agreement rent deposit period",
        "Dues & Violations": "outstanding dues rent payment violation subletting damage",
        "Demands Made": "demand vacate pay remove comply deadline",
        "Compliance Timeline": "days deadline notice comply failing legal proceedings",
    },
    "unknown": {
        "Document Overview": "main subject matter purpose",
        "Key Entities": "names parties organizations involved",
        "Key Facts": "important facts dates amounts",
    }
}


def _format_evidence_block(evidence_map: dict) -> str:
    """Format retrieved evidence as a structured block for the LLM prompt."""
    lines = []
    for section_name, evidence_list in evidence_map.items():
        if not evidence_list:
            continue
        lines.append(f"\n### Evidence for: {section_name}")
        for ev in evidence_list:
            lines.append(f"[{ev.passage.passage_id}] (score={ev.score:.2f})")
            lines.append(ev.passage.text.strip())
            lines.append("---")
    return "\n".join(lines)


def build_generation_prompt(
    doc_type: str,
    structured_fields: dict,
    evidence_map: dict,
    operator_instructions: str = "",
) -> str:

    fields_summary = json.dumps(structured_fields, indent=2, default=str)
    evidence_block = _format_evidence_block(evidence_map)

    operator_note = ""
    if operator_instructions:
        operator_note = f"""
OPERATOR STYLE INSTRUCTIONS (learned from prior edits — follow these carefully):
{operator_instructions}
"""

    prompt = f"""You are a legal document analyst. Your task is to produce a structured case fact summary
based ONLY on the evidence passages and structured fields provided below.

RULES:
1. Every factual claim MUST be supported by a specific evidence passage. Cite as [passage_id].
2. If information is unclear or marked [ILLEGIBLE]/[UNCLEAR], say so explicitly — do NOT guess.
3. If a key field is missing from the documents, note it as "NOT FOUND IN DOCUMENTS".
4. Do not add legal opinions, recommendations, or information not in the source material.
5. Structure your response exactly as the section headers below.
{operator_note}

DOCUMENT TYPE: {doc_type}

STRUCTURED FIELDS EXTRACTED:
{fields_summary}

RETRIEVED EVIDENCE PASSAGES:
{evidence_block}

Now produce a grounded case fact summary with these exact sections:
"""

    sections = DRAFT_SECTION_QUERIES.get(doc_type, DRAFT_SECTION_QUERIES["unknown"])
    for section_name in sections:
        prompt += f"\n## {section_name}\n[Write grounded content here with citations like [passage_id]]\n"

    prompt += "\n\nIMPORTANT: End your response with a line: GROUNDING_ASSESSMENT: X/10 (where X reflects how well the output is supported by evidence)"

    return prompt


# ---------------------------------------------------------------------------
# Gemini LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt: str, system: str = "") -> str:
    """
    Call Google Gemini API (gemini-2.0-flash).

    Requires:
      pip install google-generativeai
      export GEMINI_API_KEY=AIza...

    Falls back gracefully if the key is missing or the package is not installed.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return (
            "[GEMINI_API_KEY not set]\n\n"
            "Set it with: export GEMINI_API_KEY=AIza...\n"
            "Then re-run the pipeline."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system if system else (
                "You are a legal document analyst. Follow all instructions precisely."
            ),
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000,
                temperature=0.2,   # low temperature — we want factual, grounded output
            ),
        )

        return response.text

    except ImportError:
        return (
            "[google-generativeai not installed]\n\n"
            "Run: pip install google-generativeai\n"
            "Then re-run the pipeline."
        )
    except Exception as e:
        return f"[GEMINI_ERROR: {str(e)[:150]}]\n\nCheck your API key and network connection."


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def _parse_llm_sections(raw: str, section_names: list[str]) -> dict[str, str]:
    """Parse the LLM's section-by-section response into a dict."""
    result = {}
    for name in section_names:
        pattern = rf"##\s*{re.escape(name)}\s*\n(.*?)(?=\n##\s|\nGROUNDING_ASSESSMENT:|$)"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            result[name] = match.group(1).strip()
        else:
            result[name] = "[Section not generated]"
    return result


def _extract_grounding_score(raw: str) -> float:
    match = re.search(r'GROUNDING_ASSESSMENT:\s*(\d+)/10', raw)
    if match:
        return int(match.group(1)) / 10.0
    return 0.5  # default if not found


def _build_evidence_refs(section_content: str, evidence_map: dict) -> list[dict]:
    """Find which passages are actually cited in the section content."""
    cited = re.findall(r'\[([^\]]+chunk\d+[^\]]*)\]', section_content)
    refs = []
    seen = set()
    for passage_id in cited:
        if passage_id in seen:
            continue
        seen.add(passage_id)
        for ev_list in evidence_map.values():
            for ev in ev_list:
                if ev.passage.passage_id == passage_id:
                    refs.append({
                        "passage_id": passage_id,
                        "doc_id": ev.passage.doc_id,
                        "excerpt": ev.passage.text[:120] + "...",
                        "score": round(ev.score, 3),
                    })
    return refs


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_draft(
    extracted_doc,
    retrieval_store,
    operator_instructions: str = "",
    doc_type_override: Optional[str] = None,
) -> GeneratedDraft:
    """
    Full pipeline: retrieve evidence → build prompt → call Gemini → parse → return draft.
    """
    import hashlib

    doc_type = doc_type_override or extracted_doc.doc_type
    section_queries = DRAFT_SECTION_QUERIES.get(doc_type, DRAFT_SECTION_QUERIES["unknown"])

    # Retrieve evidence per section
    evidence_map = retrieval_store.retrieve_for_draft(section_queries)

    # Build prompt
    prompt = build_generation_prompt(
        doc_type=doc_type,
        structured_fields=extracted_doc.structured_fields,
        evidence_map=evidence_map,
        operator_instructions=operator_instructions,
    )

    # Call Gemini
    raw_response = call_llm(prompt)

    # Parse sections
    section_contents = _parse_llm_sections(raw_response, list(section_queries.keys()))
    overall_grounding = _extract_grounding_score(raw_response)

    # Build DraftSection objects
    sections = []
    for section_name, content in section_contents.items():
        evidence_refs = _build_evidence_refs(content, evidence_map)
        section_grounding = min(1.0, len(evidence_refs) * 0.25) if content != "[Section not generated]" else 0.0
        sections.append(DraftSection(
            section_name=section_name,
            content=content,
            supporting_passages=evidence_refs,
            grounding_score=section_grounding,
        ))

    doc_title_map = {
        "case_intake": "Case Fact Summary",
        "title_deed": "Title Review Summary",
        "legal_notice": "Notice Summary Memo",
        "unknown": "Document Summary",
    }

    draft_id = "draft_" + hashlib.md5(
        (extracted_doc.doc_id + datetime.utcnow().isoformat()).encode()
    ).hexdigest()[:8]

    return GeneratedDraft(
        draft_id=draft_id,
        doc_ids=[extracted_doc.doc_id],
        draft_type=doc_type,
        title=f"{doc_title_map.get(doc_type, 'Summary')} — {extracted_doc.doc_id}",
        sections=sections,
        overall_grounding_score=overall_grounding,
        operator_instructions=operator_instructions,
        raw_llm_response=raw_response,
    )
