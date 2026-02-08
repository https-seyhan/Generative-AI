# LLM + Machine Learning for Three‑Class Financial Crime Surveillance (Reducing False Positives)

## Objective
Maximise detection of true crime (Recall on Class 2) while minimising false alerts (False Positives on Class 0).

Classes:
- 0: Normal (no alert)
- 1: Suspicious (monitor / soft review)
- 2: Likely Financial Crime (escalate / SAR candidate)

---

## Hybrid Architecture

Narrative + KYC + Transaction Data
→ LLM (extract behavioural & legitimacy indicators)
→ Embeddings (semantic similarity)
→ LightGBM Multiclass Model (probabilities)
→ Thresholding (risk-based decision, not argmax)
→ Alert Queue

Key idea: LLM performs context interpretation. ML performs calibrated decisioning.

---

## LLM Feature Extraction (Concept)

Extract both risk and legitimacy indicators:
- structuring_risk
- mule_account_risk
- offshore_risk
- rapid_movement_of_funds
- third_party_usage
- urgency
- suspicious_entities
- legitimate_income_pattern
- family_relationship_indicator
- business_activity_indicator
- one_off_event_indicator
- historical_consistency

False positives occur when models learn only risk signals without legitimate explanations.

---

## Use Probabilities, Not Classes

Do NOT use:
    model.predict(X)

Use:
    probs = model.predict_proba(X)
    P0, P1, P2 = probs

---

## Threshold‑Based Decisioning

Example operational thresholds:

if P2 >= 0.70:
    HARD ALERT (investigator escalation)

elif P2 >= 0.40 or P1 >= 0.60:
    SOFT ALERT (monitoring queue)

else:
    NO ALERT

This controls investigation volume and reduces false positives by 60–85%.

---

## Cost‑Sensitive Training

AML errors are not equal.

False Positive → investigator cost
False Negative → regulatory risk

Example weights:

class_weights = {
    0: 1,
    1: 3,
    2: 8
}

model = LGBMClassifier(objective="multiclass", num_class=3, class_weight=class_weights)

---

## Correct Evaluation Metrics

Do not optimise accuracy.

Primary metric:
False Positive Rate (FPR) on Class 0

FPR = Normal predicted as (1 or 2) / Total Normal

Typical deployment targets:
FPR < 5%
Recall(Class 2) > 80%

---

## Cascaded Model (Advanced)

Model A: Risk Detector
Model B: Legitimacy Filter

Pipeline:
Risk Model → flags suspicious → Legitimacy Model filters normal behaviour → Final alert

This significantly reduces alert noise.

---

## Why LLM Helps

Traditional ML:
amount, frequency, country

LLM extracts:
purpose, relationship, intent, narrative consistency

Financial crime detection is fundamentally an intent detection problem.

---

## Practical Outcome

Goal is not classification accuracy.

Goal:
Prioritise investigator attention while satisfying regulatory expectations.

LLM + ML hybrid works because it reduces unnecessary alerts while preserving true detections.
