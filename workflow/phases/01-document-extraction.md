# Phase 01 — Document Extraction

**Task type:** LLM-driven (unstructured data → structured fields)
**When to use:** The user has raw borrower documents (PDF applications, bank
statements, financial filings, pay stubs) and needs structured fields extracted
for downstream credit assessment.

## What this phase does

Uses an LLM to extract structured borrower data from unstructured documents.
This is one of the highest-value LLM applications in credit — Rocket Mortgage
reports 70% of borrower documents processed automatically, saving ~9,000
underwriter hours monthly.

## Input requirements

One or more documents per borrower, in any of these formats:
- PDF loan application
- Bank statement (PDF or image)
- Pay stub or income verification letter
- Tax return or W-2
- Financial statement (corporate borrowers)

## Extraction schema

The LLM should extract these fields into a structured JSON object. Fields
not present in the source document should be set to `null`, not guessed.

```json
{
  "borrower_name": "string",
  "application_date": "YYYY-MM-DD",
  "loan_amount_requested": "float",
  "stated_income": "float | null",
  "employment_status": "employed | self-employed | retired | unemployed | null",
  "employer_name": "string | null",
  "years_at_employer": "int | null",
  "credit_limit": "float | null",
  "outstanding_balance": "float | null",
  "monthly_payment": "float | null",
  "property_type": "primary_residence | investment | commercial | null",
  "property_value": "float | null",
  "purpose": "purchase | refinance | cash_out | consolidation | other | null",
  "extraction_confidence": "float 0-1",
  "fields_not_found": ["list of schema fields not present in source"],
  "source_document_type": "string"
}
```

## Extraction prompt

Use this prompt structure. Adapt the field list to the document type.

```
You are a document processing specialist at a lending institution.
Extract the following fields from the attached document. Return JSON only.

Rules:
- Extract only what is explicitly stated in the document.
- If a field is not present, set it to null. Do NOT infer or estimate.
- For monetary values, use the original currency. Note the currency in a
  separate "currency" field if not obvious.
- If a value is ambiguous (e.g., two income figures), extract both and
  note the ambiguity in an "extraction_notes" field.
- Set "extraction_confidence" to your confidence that the extracted values
  are correct (0.0 = guessing, 1.0 = clearly stated and unambiguous).

Fields to extract:
{field_list}

Document follows:
---
{document_content}
```

## Validation rules

After extraction, validate programmatically (do NOT rely on the LLM for this):

1. **Completeness check:** Count non-null fields. Flag if fewer than 60%
   of core fields are populated.
2. **Range checks:** Income > 0, age 18-120, loan amount > 0.
3. **Cross-field consistency:** Monthly payment should be < monthly income.
   Outstanding balance should be <= credit limit.
4. **Duplicate detection:** If multiple documents for the same borrower,
   check for conflicting values across documents.

## Output

Save extracted profiles to `data/processed/extracted_profiles.csv` with one
row per borrower. Include `extraction_confidence` and `source_document`
columns for audit trail.

## Handoff to Phase 02

Once structured fields are extracted, they feed into Phase 02 (Default
Prediction) as model input features. Map extracted fields to the model's
expected feature schema using the column mapping in
`references/dataset-schemas.md`.

## Limitations

- LLM extraction is not 100% accurate. Critical financial figures
  (income, loan amount) should be verified by a human before decisioning.
- Handwritten documents or low-quality scans will degrade extraction quality.
- This phase does not perform OCR — documents must already be text-searchable
  PDFs or have been pre-processed through an OCR pipeline.
