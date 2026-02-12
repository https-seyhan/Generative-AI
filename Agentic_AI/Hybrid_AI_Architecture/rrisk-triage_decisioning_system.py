Designing a **risk-triage decisioning system**, not a chatbot.
The architecture should behave like a *junior analyst + rules engine + statistical model + supervisor* — and the key business constraint you stated is critical:

> The cost of a false-positive **escalate** is higher than a false-negative.

That changes the entire modelling strategy.
You **do NOT want a symmetric classifier**.
You want a **selective classifier with abstention** (aka *classification with reject option*).

Your “unclear” class is not a class — it is a **decision state**.

---

# 1) Target behaviour (what the system must actually do)

You are not predicting labels.

You are making **operational decisions**:

| Decision | Meaning                    | Cost                       |
| -------- | -------------------------- | -------------------------- |
| Decline  | Safe / low risk            | Low cost                   |
| Escalate | High confidence suspicious | Very high operational cost |
| Unclear  | Human analyst review       | Moderate cost              |

Therefore:

> Escalate must only occur when evidence is strong and corroborated across independent signals.

That is how you reduce false positives.

So we will not let a single model escalate anything.
Escalate must be a **multi-signal consensus event**.

---

# 2) Correct Architecture (Agentic Hybrid AI)

This is the proper structure:

```
                    ┌──────────────┐
Input Case  ───────▶│ LLM Analyst  │
                    └──────┬───────┘
                           │ structured signals
                           ▼
                    ┌──────────────┐
                    │ Feature Builder│
                    └──────┬───────┘
                           │
           ┌───────────────┼─────────────────┐
           ▼               ▼                 ▼
  Behaviour ML      Network ML         Rule Engine
  (tabular)        (graph features)     (hard logic)

           └───────────────┬─────────────────┘
                           ▼
                   Decision Calibrator
                    (Risk Aggregator)
                           ▼
                    Decision Policy
              (Decline / Escalate / Unclear)
```

This is an **agentic workflow**, not just an ensemble.

The LLM is an *investigator*, not the decision maker.

---

# 3) Role of the LLM (very important)

The LLM should NEVER directly output *escalate*.

The LLM’s job:

**Convert messy case text into risk evidence signals.**

Example input:

* transaction notes
* customer communication
* payment description
* KYC explanation
* onboarding responses

### LLM Output (structured)

You force it into JSON:

```json
{
  "structuring_score": 0.82,
  "mule_account_probability": 0.71,
  "intent_evasion": 0.63,
  "documentation_consistency": 0.34,
  "explanation_quality": 0.28,
  "cooperation_level": 0.45
}
```

These are not decisions.
They are **latent behavioural indicators**.

Why?

Because LLMs are extremely good at:
• deception detection patterns
• linguistic hedging
• excuse narratives
• evasion language

But they are terrible at:
• calibrated probabilities
• operational risk thresholds

So the LLM becomes a **feature generator**.

---

# 4) Machine Learning Models (what you actually train)

You need *two* models.

## Model 1 — Behavioural Risk Model (Tabular ML)

Algorithm:
**LightGBM or XGBoost (not deep learning)**

Inputs:

* transaction velocity
* circular transfers
* new payee frequency
* night activity ratio
* account age
* geo volatility
* LLM behavioural features (from above)

Output:

```
P(High_Risk)
```

---

## Model 2 — Network Risk Model (Graph ML)

You are missing this — and it is why false positives happen.

Fraud rarely exists in isolation.

Create graph features:

Nodes:

* accounts
* devices
* phone numbers
* emails
* IPs
* beneficiaries

Edges:

* transfers
* shared devices
* shared contact info

Then compute:

| Feature                  | Meaning                  |
| ------------------------ | ------------------------ |
| PageRank risk            | proximity to known fraud |
| shared beneficiary count | mule rings               |
| 2-hop fraud connection   | organised activity       |
| clustering coefficient   | network behaviour        |

Train another LightGBM:

```
P(Network_Risk)
```

---

# 5) Why you are getting false positive escalates

Currently your system is likely doing:

> suspicious behaviour → escalate

But suspicious ≠ fraud.

You are detecting **anomalies**, not **malice**.

Typical false positive:

* small business cash flow
* family transfers
* shared households
* remittance patterns
* cultural financial behaviour

A single model cannot distinguish that.

You need **cross-evidence validation**.

---

# 6) The Decision Calibrator (this is the missing piece)

We now combine 3 independent evidences:

```
Behaviour Risk
Network Risk
LLM Intent Risk
```

We do NOT majority vote.

We build a calibrated risk:

[
FinalRisk =
0.45 * Behaviour +
0.35 * Network +
0.20 * LLMIntent
]

Then apply **selective classification thresholds**.

---

# 7) Thresholding Strategy (how you reduce false positives)

Instead of 3-class classification:

You use **confidence bands**.

| Final Risk               | Decision        |
| ------------------------ | --------------- |
| < 0.35                   | Decline         |
| 0.35 – 0.72              | Unclear (human) |
| > 0.72 AND corroboration | Escalate        |

Critical rule:

> Escalate only if TWO independent models are high.

Example:

| Behaviour | Network | LLM  | Result                   |
| --------- | ------- | ---- | ------------------------ |
| 0.85      | 0.20    | 0.90 | ❌ Unclear (NOT escalate) |
| 0.80      | 0.78    | 0.40 | ✅ Escalate               |
| 0.60      | 0.62    | 0.65 | Unclear                  |
| 0.30      | 0.25    | 0.20 | Decline                  |

This alone will drop false-positive escalates dramatically.

---

# 8) Implement Abstention (your “unclear” class)

Technically this is:

**Selective prediction using conformal risk control**

Instead of predicting a class:

Model predicts:

```
risk score + uncertainty
```

If prediction confidence < threshold → send to human.

This is not optional.
Regulated decisioning systems must include this.

---

# 9) Agentic Layer (why this becomes “Agentic AI”)

Now we add agents:

### Planner Agent

Decides what to investigate.

### Investigator Agent (LLM)

Extracts behavioural signals from documents.

### Evidence Agent

Queries transaction DB + graph DB.

### Validator Agent

Checks if evidence corroborates.

### Decision Agent

Applies policy thresholds.

So escalation becomes:

> A supervised multi-step reasoning process, not a model output.

---

# 10) Practical Training Strategy

Do **NOT** train on labels:

* escalate
* decline

Train on:

```
confirmed fraud
confirmed clean
```

Then let the policy layer map to operations.

Why?

Because “escalate” depends on analyst behaviour, staffing, and risk appetite — not ground truth.

---

# 11) Key metric you should optimise

Not accuracy.

Not F1.

You want:

### Escalate Precision

[
Precision_{Escalate} = \frac{TrueFraudEscalates}{AllEscalates}
]

and

### Review Yield

[
FraudFoundInReview / TotalReviews
]

Your system is a **workflow optimizer**, not a classifier.

---

# 12) What will reduce false positive escalates the most

In order of impact:

1. Network graph features (biggest effect)
2. Two-model corroboration rule
3. Abstention band (unclear)
4. LLM used as feature extractor (not classifier)
5. Probability calibration (Platt scaling / isotonic)

---

If you want, I can next give you:
• the exact JSON schema for the LLM investigator
• LightGBM training feature list
• and the decision policy pseudocode for production deployment.
