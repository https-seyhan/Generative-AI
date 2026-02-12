**production-oriented specification** you can implement immediately.
The goal is: the LLM never makes the decision — it produces *auditable behavioural evidence* that becomes ML features.

We will define three artifacts:

1. LLM Investigator JSON schema (strict, machine-parsable)
2. LightGBM feature set (tabular + graph + LLM features)
3. Decision policy pseudocode (how “escalate / decline / unclear” is actually produced)

---

# 1) LLM Investigator — Exact JSON Schema

You must **force structured output**.
Do *not* accept free-text. Reject any response that fails validation.

The LLM is acting as a **behavioural analyst** performing linguistic risk assessment.

### Required Output Rules

* All numeric scores ∈ [0,1]
* No missing fields
* No prose explanations outside `rationale_summary`
* Deterministic temperature (≤0.2)

---

## JSON Schema (Draft 2020-12 compatible)

```json
{
  "type": "object",
  "required": [
    "case_id",
    "linguistic_risk",
    "intent_assessment",
    "behavioural_indicators",
    "documentation_assessment",
    "consistency_checks",
    "overall_assessment"
  ],
  "properties": {

    "case_id": { "type": "string" },

    "linguistic_risk": {
      "type": "object",
      "required": [
        "evasion_language",
        "overjustification",
        "scripted_phrasing",
        "urgency_pressure",
        "third_party_control"
      ],
      "properties": {
        "evasion_language": { "type": "number", "minimum": 0, "maximum": 1 },
        "overjustification": { "type": "number", "minimum": 0, "maximum": 1 },
        "scripted_phrasing": { "type": "number", "minimum": 0, "maximum": 1 },
        "urgency_pressure": { "type": "number", "minimum": 0, "maximum": 1 },
        "third_party_control": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "intent_assessment": {
      "type": "object",
      "required": [
        "mule_account_probability",
        "deception_probability",
        "knowledge_of_transactions",
        "beneficiary_relationship_plausibility"
      ],
      "properties": {
        "mule_account_probability": { "type": "number", "minimum": 0, "maximum": 1 },
        "deception_probability": { "type": "number", "minimum": 0, "maximum": 1 },
        "knowledge_of_transactions": { "type": "number", "minimum": 0, "maximum": 1 },
        "beneficiary_relationship_plausibility": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "behavioural_indicators": {
      "type": "object",
      "required": [
        "structuring_likelihood",
        "circular_payment_explanation_quality",
        "transaction_purpose_clarity",
        "cooperation_level",
        "response_specificity"
      ],
      "properties": {
        "structuring_likelihood": { "type": "number", "minimum": 0, "maximum": 1 },
        "circular_payment_explanation_quality": { "type": "number", "minimum": 0, "maximum": 1 },
        "transaction_purpose_clarity": { "type": "number", "minimum": 0, "maximum": 1 },
        "cooperation_level": { "type": "number", "minimum": 0, "maximum": 1 },
        "response_specificity": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "documentation_assessment": {
      "type": "object",
      "required": [
        "document_consistency",
        "document_authenticity_suspected",
        "information_completeness"
      ],
      "properties": {
        "document_consistency": { "type": "number", "minimum": 0, "maximum": 1 },
        "document_authenticity_suspected": { "type": "number", "minimum": 0, "maximum": 1 },
        "information_completeness": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "consistency_checks": {
      "type": "object",
      "required": [
        "internal_story_consistency",
        "transaction_vs_explanation_alignment",
        "timeline_plausibility"
      ],
      "properties": {
        "internal_story_consistency": { "type": "number", "minimum": 0, "maximum": 1 },
        "transaction_vs_explanation_alignment": { "type": "number", "minimum": 0, "maximum": 1 },
        "timeline_plausibility": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "overall_assessment": {
      "type": "object",
      "required": [
        "llm_intent_risk",
        "confidence",
        "rationale_summary"
      ],
      "properties": {
        "llm_intent_risk": { "type": "number", "minimum": 0, "maximum": 1 },
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "rationale_summary": { "type": "string", "maxLength": 500 }
      }
    }
  }
}
```

This schema turns the LLM into a **behavioural scoring instrument** rather than a classifier.

---

# 2) LightGBM Training Feature List

You will train **two LightGBM models**:

• Behaviour Model
• Network Model

---

## A. Core Transaction Behaviour Features

**Velocity / Amount**

* tx_count_1d
* tx_count_7d
* outgoing_ratio_7d
* avg_tx_amount_7d
* median_tx_amount
* max_tx_amount_30d
* amount_stddev_7d
* burstiness_index (Poisson deviation)

**Account Lifecycle**

* account_age_days
* days_since_kyc
* first_tx_after_open_hours
* dormant_reactivation_flag

**Counterparty Behaviour**

* new_payee_count_7d
* new_payee_ratio
* unique_beneficiaries_30d
* beneficiary_reuse_ratio
* shared_beneficiary_accounts

**Temporal Behaviour**

* night_tx_ratio
* weekend_tx_ratio
* tx_hour_entropy
* session_count_24h

**Circularity**

* return_payment_ratio
* 2hop_return_flag
* 3hop_return_flag
* circular_value_percentage

**Geo / Device**

* unique_devices_7d
* ip_country_count_7d
* geo_velocity_km_hr
* vpn_probability
* device_change_rate

---

## B. Graph / Network Features (CRITICAL)

From graph DB (Neo4j recommended):

* pagerank_risk
* fraud_neighbor_count_1hop
* fraud_neighbor_count_2hop
* shared_device_degree
* shared_phone_degree
* shared_email_degree
* mule_cluster_membership_score
* connected_component_size
* fraud_ring_probability
* beneficiary_centrality
* network_entropy
* community_risk_score

This alone will cut false escalates massively.

---

## C. LLM Features (derived from schema)

Flatten JSON → numeric features:

Examples:

```
llm_evasion_language
llm_overjustification
llm_scripted_phrasing
llm_mule_probability
llm_deception_probability
llm_structuring_likelihood
llm_cooperation_level
llm_story_consistency
llm_doc_consistency
llm_intent_risk
llm_confidence
```

Important:

> Do NOT feed the rationale text to the model. Only numeric signals.

---

# 3) Decision Policy Pseudocode (Production)

This is the most important part.

We are implementing a **corroborated escalation policy**.

---

## Step 1 — Model Outputs

```
B = BehaviourModel.predict_proba(case)
N = NetworkModel.predict_proba(case)
L = LLM_intent_risk
C = LLM_confidence
```

---

## Step 2 — Calibrate Probabilities

(You must apply isotonic or Platt scaling)

```
B_cal = calibrate_behaviour(B)
N_cal = calibrate_network(N)
```

---

## Step 3 — Aggregate Risk

```
final_risk =
    0.45 * B_cal +
    0.35 * N_cal +
    0.20 * L
```

---

## Step 4 — Corroboration Logic (False-Positive Protection)

```
high_behaviour = B_cal > 0.75
high_network   = N_cal > 0.70
high_llm       = (L > 0.70 and C > 0.60)
```

---

## Step 5 — Decision Engine

```
IF final_risk < 0.35:
    DECISION = "decline"

ELIF final_risk >= 0.72 AND
     ((high_behaviour and high_network) OR
      (high_network and high_llm)):

    DECISION = "escalate"

ELSE:
    DECISION = "unclear"
```

This is the key rule:

> Escalation requires independent evidence agreement.

One model alone cannot escalate.

---

# 4) Why this reduces false positives

Most false escalates come from:

* high anomaly but legitimate activity
* unusual but culturally normal transfers
* small business payment loops

Behaviour model alone flags these.

But those **do not form fraud networks**.

The network corroboration gate prevents escalation.

---

# 5) What to log (very important for auditors)

Store per case:

```
B_cal
N_cal
L
final_risk
triggered_rules
decision
feature_contributions (SHAP values)
LLM_rationale_summary
```

This makes the system explainable to regulators and internal risk committees.

---

If you want next, I can design the **prompt template for the investigator LLM** that reliably produces this schema without hallucinations (that part is actually harder than the modelling).
