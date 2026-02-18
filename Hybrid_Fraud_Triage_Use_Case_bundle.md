
# Use Case: Capacity‑Constrained Dual‑Threshold Triage in a Hybrid LLM + ML Fraud Detection System

## 1. Business Context
A financial services organization processes high‑volume daily cases (payments, applications, or entity changes). Traditional rule engines generate excessive false positives (FPs), overwhelming investigators and degrading customer experience.

Objective:
- Automate clear non‑risk cases (auto‑decline)
- Automatically act on very high‑risk cases (auto‑escalate)
- Route ambiguous cases to human investigators (unclear)
- Maintain investigator workload within operational capacity

This system implements **selective classification** using a hybrid architecture:
- LLM: semantic investigator → extracts structured risk signals
- LightGBM: calibrated decision engine → estimates fraud likelihood
- Dual thresholds: operational trust policy
- Queueing control: keeps human review stable

---

## 2. Actors
| Actor | Role |
|------|------|
| Customer/System | Generates a case/transaction |
| LLM Investigator | Converts unstructured evidence to structured signals |
| ML Risk Engine | Produces calibrated fraud probability |
| Decision Policy | Applies dual thresholds |
| Human Investigator | Resolves ambiguous cases |
| Risk Operations Manager | Monitors capacity and KPIs |

---

## 3. Data Flow
1. Case created
2. LLM reads documents, notes, metadata
3. LLM outputs structured risk signals
4. ML model produces probability p = P(fraud | signals)
5. Decision engine compares p to TL and TH
6. Case is auto‑cleared, reviewed, or escalated

---

## 4. LLM Investigator Output Schema (Example)
```json
{
  "entity_linkage_score": 0.82,
  "director_overlap": true,
  "address_similarity": 0.74,
  "circular_transactions": 0.61,
  "velocity_anomaly": 0.47,
  "tax_registration_gap_days": 18,
  "narrative_risk_summary": "New entity shares director and service address with recently liquidated firm."
}
```

The LLM does **not** make the decision.  
It produces **features**.

---

## 5. ML Risk Engine
Model: LightGBM classifier

Inputs:
- LLM structured signals
- Behavioral features
- Transaction patterns
- Historical compliance indicators

Output:
p = calibrated probability of fraud (0–1)

### Calibration
Use isotonic regression:
```python
from sklearn.calibration import CalibratedClassifierCV
cal_model = CalibratedClassifierCV(lgbm, method="isotonic", cv=5)
cal_model.fit(X_train, y_train)
p = cal_model.predict_proba(X_case)[1]
```

Calibration ensures:
> A score of 0.80 ≈ 80% likelihood of fraud.

---

## 6. Dual‑Threshold Decision Policy

Define:
TL = low‑risk threshold  
TH = high‑risk threshold

Decision regions:

| Probability | Action |
|------|------|
| p ≤ TL | Decline (auto clear) |
| TL < p < TH | Unclear → Human review |
| p ≥ TH | Escalate |

Pseudocode:
```
if p >= TH:
    decision = "ESCALATE"
elif p <= TL:
    decision = "DECLINE"
else:
    decision = "UNCLEAR"
```

---

## 7. Why Dual Thresholds Reduce False Positives
False positives occur mainly in the **ambiguity band** (mid‑probability scores).  
The middle region contains cases with partial fraud indicators.

Instead of forcing an incorrect automated decision:
→ route them to humans.

Result:
- FPs drop sharply
- Recall preserved

---

## 8. Capacity‑Constrained Thresholding

Let:
λ = daily incoming cases
μ = reviews per analyst per day
c = number of analysts

Human processing capacity:
C = c × μ

System stability condition:
λ_h < C

Where:
λ_h = r × λ  
r = fraction of cases in the unclear band

Therefore:
r_max = C / λ

Thresholds must satisfy:
P(TL < p < TH) ≤ r_max

---

## 9. Threshold Optimization Algorithm
1. Compute calibrated probabilities on validation set
2. Determine allowable review fraction (r_max)
3. Search threshold pairs producing review rate ≤ r_max
4. Choose pair minimizing cost

Example cost:
Cost = 10 × FP + 50 × FN

---

## 10. Weekly Adaptive Calibration
Fraud patterns drift → score distribution shifts

Weekly process:
1. Recompute probability distribution
2. Re‑estimate review rate
3. Adjust TL and TH
4. Deploy new thresholds

This is **adaptive triage calibration**.

---

## 11. KPIs
| KPI | Purpose |
|------|------|
| False Positive Rate | Customer impact |
| Fraud Capture Rate | Risk control |
| Review Queue Length | Operational health |
| Average Handling Time | Staffing |
| SLA Compliance | Service quality |

---

## 12. Acceptance Criteria
- Review queue stable (no growth trend)
- FPR < 1%
- ≥ 80% of cases auto‑decided
- Analyst workload within daily capacity

---

## 13. Key Insight
The system is not a fraud classifier.

It is a **risk triage engine** combining:
AI certainty + human judgment + operational capacity.
