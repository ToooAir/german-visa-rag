"""
Statutory thresholds for German visa eligibility.

These values are set by regulation and updated periodically.
Each constant includes its effective date and the previous year's value
so that future maintainers know when and how to update them.

Update checklist (typically each January):
  1. Verify new thresholds at https://www.make-it-in-germany.com or BMI announcements.
  2. Move current values to the "Previous year" comment.
  3. Update the constant values and the effective date comment.
  4. Run `python -m eval.state_tag_evaluator` to confirm no F1 regression.
"""

# ── EU Blue Card salary thresholds ────────────────────────────────────────────
# Source: § 18g AufenthG + annual BMI regulation (linked to pension insurance ceiling)
# Effective: 2026-01-01
# Previous year 2025: shortage €43,759.80 / general €48,300.00

BLUE_CARD_SALARY_SHORTAGE_2026: float = 45_934.20
"""Shortage occupations (IT, Engineering, STEM, Natural Sciences, Healthcare).
Also applies to recent graduates (≤ 3 years post-graduation) regardless of occupation."""

BLUE_CARD_SALARY_GENERAL_2026: float = 50_700.00
"""General occupations — applies when role is not a shortage occupation
and applicant is not a recent graduate."""

BLUE_CARD_SALARY_GRADUATE_2026: float = 45_934.20
"""Recent graduates: applicants within 3 years of their graduation date.
Same numeric value as shortage threshold (both track the same BMI ceiling fraction)
but represents a separate legal pathway."""

# ── Chancenkarte financial proof threshold ─────────────────────────────────────
# Source: § 20 AufenthG — blocked account (Sperrkonto) or equivalent
# Effective: 2026-01-01  (same as student visa, tied to BAföG Bedarfssatz)
# Previous year 2025: €13,068.00

CHANCENKARTE_FINANCIAL_PROOF_2026: float = 13_092.00

# ── Student Visa financial proof threshold ─────────────────────────────────────
# Source: § 16b AufenthG — blocked account (Sperrkonto)
# Effective: 2026-01-01
# Previous year 2025: €11,208.00

STUDENT_FINANCIAL_PROOF_2026: float = 11_904.00
