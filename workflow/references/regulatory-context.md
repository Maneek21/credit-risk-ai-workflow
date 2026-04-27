# Regulatory Context Reference

## Key Regulations for AI/ML in Credit

### Equal Credit Opportunity Act (ECOA) — 15 U.S.C. § 1691

**Enforced by:** CFPB (Consumer Financial Protection Bureau)
**Implementing regulation:** Regulation B (12 CFR 1002)

**Core requirements:**
- Prohibits discrimination in any aspect of a credit transaction on the
  basis of race, color, religion, national origin, sex, marital status,
  age, receipt of public assistance, or exercise of consumer protection rights
- Requires adverse-action notices with specific reasons for denial
- Notice must be provided within 30 days of adverse action
- Reasons must be specific to the applicant, not generic

**Relevance to AI/ML models:**
- Models must not use prohibited bases as features (or proxies)
- Even facially neutral features can violate ECOA if they have disparate impact
  and are not justified by business necessity
- Adverse-action reasons must be derivable from the model's actual decision logic
  (this is where SHAP-grounded explanations become essential)

### Fair Credit Reporting Act (FCRA) — 15 U.S.C. § 1681

**Enforced by:** FTC and CFPB

**Key provisions for adverse action:**
- If a credit report was used in the decision, the notice must identify
  the credit reporting agency
- Consumer must be informed of right to obtain a free copy of the report
- Consumer must be informed of right to dispute inaccurate information
- "Risk-based pricing" notices required when terms are less favorable
  than the best available

### SR 11-7: Guidance on Model Risk Management

**Issued by:** Federal Reserve Board, OCC (2011)
**Applies to:** All banking organizations using models

**Three pillars:**
1. **Model development** — sound design, theory, and logic
2. **Model validation** — independent review, backtesting, benchmarking
3. **Model governance** — policies, controls, documentation, audit trail

**Relevance to AI/ML:**
- Models must be validated against holdout data (our Phase 02)
- Models must be monitored for performance degradation (our Phase 06)
- Model limitations must be documented and communicated
- "Effective challenge" — independent review of model methodology
- Applies to vendor models (including LLM APIs) used in credit decisions

**Key quote:** "The use of models invariably presents model risk, which is
the potential for adverse consequences from decisions based on incorrect
or misused model outputs and reports."

### Four-Fifths Rule (80% Rule)

**Source:** EEOC Uniform Guidelines on Employee Selection Procedures (1978),
29 CFR 1607.4(D)

**Application to credit:** While originally for employment, the four-fifths
rule is widely used in fair lending analysis as a screening tool for
disparate impact.

**The rule:** A selection rate for any protected group that is less than
80% of the rate for the group with the highest rate constitutes evidence
of adverse impact.

**Legal weight:** The four-fifths rule creates a rebuttable presumption.
Failing it is not automatic liability — the lender can defend with:
1. Business necessity (the factor is genuinely predictive)
2. No less discriminatory alternative exists
3. The disparity is explained by legitimate credit factors

**Our findings:**
- Classical models: XGBoost fails for "2+ minority races" (DI 0.75) on HMDA
- LLMs: All fail on education (DI 0.49–0.64); all pass on sex
- Skill-augmented GPT-4o: Education DI improved to 0.71 (still fails)

### CFPB Guidance on AI in Lending

**Key positions (as of 2024-2025):**
- Creditors cannot use "black box" AI as a defense against ECOA obligations
- Adverse-action notices must be specific and accurate regardless of
  model complexity
- "The computational method by which the creditor arrives at the basis
  for the credit decision does not change the requirements"
- CFPB has signaled increased scrutiny of AI-driven lending decisions

## International Context

### EU AI Act (2024)

- Credit scoring classified as "high-risk" AI system
- Requires transparency, human oversight, and non-discrimination
- Mandatory conformity assessments before deployment
- Effective 2026 for high-risk systems

### Basel Committee on Banking Supervision

- Principles for sound management of operational risk (2024 update)
  includes AI/ML governance
- Emphasizes model validation, ongoing monitoring, and explainability

## Practical Implications for This Workflow

| Requirement | How this workflow addresses it |
|---|---|
| Specific adverse-action reasons | Phase 05 uses SHAP-grounded reasons |
| Non-discrimination | Phase 04 computes DI ratios against four-fifths rule |
| Model validation | Phase 02 holdout testing + Phase 06 drift monitoring |
| Documentation | Every phase writes to `results/` with reproducible artifacts |
| Human oversight | SKILL.md states all decisions require human review |
| Ongoing monitoring | Phase 06 implements SR 11-7 monitoring |
| Explainability | SHAP explanations in Phase 02 + Phase 03 memo grounding |
