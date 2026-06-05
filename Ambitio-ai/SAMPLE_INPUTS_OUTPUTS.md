# Sample Inputs & Outputs

## Sample Input: `case_intake_003.txt` (messy handwriting transcription)

```
CASE INTAKE NOTES - HANDWRITTEN TRANSCRIPTION
[Transcribed from handwritten notes - accuracy not guaranteed]

Case Ref: HYD/CIVIL/2024/0087
...
- Client booked Flat No. 304, Tower B in "Serene Heights" project by ABC Constructions
- Booking amount: Rs 3,50,000 paid on 15/Mar/2021 (receipt available)
- Builder-Buyer Agreement signed: 22/Apr/2021
- Promised possession: December 2022 (as per agreement Clause 8)
- ACTUAL STATUS: Construction at ~60% as of Dec 2023 (1 year delayed already)
...
```

---

## Stage 1 Output — Structured Extraction

```json
{
  "doc_id": "doc_dfa9baef",
  "doc_type": "case_intake",
  "confidence_flags": {
    "quality_score": 80,
    "has_ocr_artifacts": false,
    "has_damage_markers": false,
    "missing_attachments": false
  },
  "warnings": [
    "OCR fix applied: pattern '\\b0(?=[a-zA-Z])' matched 1 time(s)"
  ],
  "structured_fields": {
    "client_name": "Deepak Anand Kulkarni",
    "opposing_party": "ABC Constructions Pvt Ltd",
    "nature_of_dispute": "Consumer complaint + civil suit re: apartment purchase",
    "case_ref": "HYD/CIVIL/2024/0087",
    "documents_present": [
      "Booking receipt",
      "Builder-Buyer Agreement",
      "Payment receipts (partial - 6 of 9 installments documented)",
      "Builder's Nov 2023 letter"
    ],
    "documents_missing": [
      "RERA registration certificate - NOT with client (to be obtained)",
      "Sanctioned building plan - NOT with client",
      "Completion certificate - N/A (project incomplete)"
    ],
    "all_amounts": ["Rs 3,50,000", "Rs 65,00,000 (sixty-five lakhs)", "Rs 48,00,000"],
    "next_steps": [
      "Obtain RERA registration details",
      "Verify all payment receipts",
      "Draft complaint to RERA Authority Telangana",
      "Consider consumer forum as parallel track"
    ]
  }
}
```

---

## Stage 2 Output — Retrieved Evidence

For the query: `"client name dispute RERA builder payment"`

```
[score=8.810] doc_dfa9baef_chunk3
  "Client wants: either possession OR full refund + compensation + interest
   DOCUMENTS AVAILABLE: [x] Booking receipt [x] Builder-Buyer Agreement..."

[score=6.487] doc_dfa9baef_chunk2
  "NATURE OF DISPUTE: Consumer complaint + civil suit re: apartment purchase
   FACTS: Client booked Flat No. 304, Tower B in 'Serene Heights'..."

[score=4.490] doc_dfa9baef_chunk5
  "PRELIMINARY ASSESSMENT: RERA applicable - project likely registered
   Force majeure claim: weak given 2-year delay post-COVID..."
```

---

## Stage 3 Output — Generated Draft (Markdown)

```markdown
# Case Fact Summary — doc_dfa9baef

*Generated: 2025-01-01T12:00:00 | Grounding: 75%*

## Party Identification
The client is **Deepak Anand Kulkarni** [doc_dfa9baef_chunk1], a resident of
Punjagutta, Hyderabad. The opposing party is **ABC Constructions Pvt. Ltd.**,
with registered office at Laxmi Towers, Begumpet [doc_dfa9baef_chunk2].
Case reference: HYD/CIVIL/2024/0087.

**Evidence:**
- `doc_dfa9baef_chunk1` (score 7.23): "Name: Deepak Anand Kulkarni..."
- `doc_dfa9baef_chunk2` (score 6.49): "ABC Constructions Pvt Ltd..."

## Nature of Dispute
Consumer complaint and civil suit regarding delayed possession of a
residential apartment [doc_dfa9baef_chunk2]. Client booked Flat No. 304,
Tower B in the "Serene Heights" project for Rs. 65,00,000, with possession
promised by December 2022 per Clause 8 of the Builder-Buyer Agreement.
As of December 2023, construction stands at approximately 60% [doc_dfa9baef_chunk2].

## Financial Summary
Total agreed consideration: Rs. 65,00,000 [doc_dfa9baef_chunk2].
Amount paid to date: Rs. 48,00,000 across 6 of 9 installments
[doc_dfa9baef_chunk3]. Three payment receipts are undocumented — these
should be obtained before filing.

## Document Status
Present: Booking receipt, Builder-Buyer Agreement, 6 payment receipts,
builder's force majeure letter (Nov 2023) [doc_dfa9baef_chunk3].
Missing: RERA registration certificate, sanctioned building plan,
completion certificate (N/A — project incomplete) [doc_dfa9baef_chunk3].

## Preliminary Assessment
The intake notes suggest RERA is applicable and the force majeure claim
is likely weak given the two-year delay post-COVID easing [doc_dfa9baef_chunk5].
Compensation under RERA Section 18 is noted as viable. A parallel consumer
forum track is recommended [doc_dfa9baef_chunk5].
```

---

## Stage 4 Output — After Operator Edit + Improvement

**Operator edit added**: A "Risk Flags" section and renamed "Party Identification" to "Parties Involved".

**Rules extracted by improvement system**:
```
1. Be more thorough and detailed — expand explanations and include more context
2. Always include these additional sections: Parties Involved, Risk Flags
3. Explicitly flag risks, missing documents, and ambiguities — don't minimize concerns
```

**Effect on next draft**: These rules are injected into the next generation prompt as:
```
OPERATOR STYLE INSTRUCTIONS (learned from prior edits — follow these carefully):
1. Be more thorough and detailed — expand explanations and include more context
2. Always include these additional sections: Parties Involved, Risk Flags
3. Explicitly flag risks, missing documents, and ambiguities — don't minimize concerns
```

The next draft for any `case_intake` document will automatically include a Risk Flags section and use more expansive prose.

---

## Evaluation Results

| Dimension | Score |
|-----------|-------|
| Extraction (field coverage + type accuracy) | 100% |
| Retrieval (precision@3 + top-1 accuracy) | 100% |
| Grounding (citation coverage + LLM self-score) | 87.5% |
| Improvement (rule extraction accuracy) | 100% |
| **Weighted Overall** | **98.5%** |
