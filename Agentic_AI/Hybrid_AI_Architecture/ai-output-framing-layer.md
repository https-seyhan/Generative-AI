# AI Output Framing Layer Architecture

## Overview

The AI Output Framing Layer transforms raw AI model outputs into
structured, interpretable, auditable, and operational decisions.

Without framing, AI outputs can be: - ambiguous - difficult to
operationalise - hard to audit - prone to human misinterpretation

The framing layer standardises outputs so they can be safely consumed by
investigators, analysts, or automated systems.

Typical use cases include:

-   financial crime detection
-   fraud investigation
-   phoenixing detection
-   compliance monitoring
-   operational decision automation

------------------------------------------------------------------------

# Architecture Principles

### Deterministic Decisions

All AI outputs must resolve to a fixed decision class.

Example:

ESCALATE\
DECLINE\
UNCLEAR

### Structured Signals

Model reasoning must be converted into machine-readable signals.

### Calibration

LLM outputs must be calibrated by statistical models.

### Interpretability

Outputs must be understandable by human analysts.

### Auditability

All decisions must include governance metadata.

------------------------------------------------------------------------

# High-Level Architecture

Input Case Data ↓ LLM Investigator (signal generator) ↓ Signal
Extraction Layer ↓ ML Calibration Model ↓ Decision Policy Engine ↓ AI
Output Framing Layer ↓ Human or System Consumer

------------------------------------------------------------------------

# Layer Responsibilities

## 1. LLM Investigator

The LLM extracts semantic signals from structured and unstructured data.

Example signals:

-   director_multiple_liquidations
-   address_reuse
-   rapid_company_recreation
-   abn_cancellation_before_liquidation

The LLM should output **structured signals instead of free text**.

------------------------------------------------------------------------

# Signal Extraction Layer

Signals are normalised into machine readable format.

Example JSON:

{ "case_id": "C12345", "signals": { "director_multiple_liquidations":
true, "address_reuse": true, "rapid_reincorporation": false } }

Signals are stored for:

-   training
-   audit
-   explainability

------------------------------------------------------------------------

# ML Calibration Layer

Signals are passed to a statistical risk model.

Typical models:

-   LightGBM
-   Logistic Regression
-   Random Forest

Example output:

fraud_probability = 0.81

This represents similarity to known fraud cases.

------------------------------------------------------------------------

# Decision Policy Engine

Probability scores are converted into decisions.

Example thresholds:

Score \> 0.75 → ESCALATE\
Score 0.35--0.75 → UNCLEAR\
Score \< 0.35 → DECLINE

Example Python logic:

``` python
def decision_policy(score):

    if score >= 0.75:
        return "ESCALATE"

    elif score <= 0.35:
        return "DECLINE"

    else:
        return "UNCLEAR"
```

------------------------------------------------------------------------

# AI Output Framing Layer

The final decision is packaged for humans and machines.

Example:

{ "case_id": "C12345", "decision": "ESCALATE", "risk_score": 0.81,
"confidence": "HIGH", "signals": \[ "director_multiple_liquidations",
"address_reuse" \], "recommended_action": "Send to investigation team" }

------------------------------------------------------------------------

# Human Interpretation Layer

Example investigator output:

## CASE DECISION

Outcome: ESCALATE Risk Score: 0.81 Confidence: HIGH

## Key Signals

• Director linked to multiple failed entities\
• Address reused across companies

## Recommended Action

Assign to investigation team

------------------------------------------------------------------------

# Governance and Auditability

Production AI systems must record metadata.

Example:

{ "ml_model_version": "risk_model_v4", "llm_model_version":
"investigator_v1.3", "decision_policy": "dual_threshold_v2",
"timestamp": "2026-03-10T10:21:00" }

This supports:

-   regulatory review
-   reproducibility
-   audit investigations

------------------------------------------------------------------------

# Example End-to-End Flow

Case Data\
↓\
LLM Investigator\
↓\
Structured Signals\
↓\
ML Risk Model\
↓\
Probability Score\
↓\
Decision Policy\
↓\
AI Output Framing Layer\
↓\
Investigator Dashboard / API

------------------------------------------------------------------------

# Implementation Example

``` python
def frame_ai_output(case_id, signals, score):

    if score >= 0.75:
        decision = "ESCALATE"
        confidence = "HIGH"

    elif score <= 0.35:
        decision = "DECLINE"
        confidence = "HIGH"

    else:
        decision = "UNCLEAR"
        confidence = "MEDIUM"

    return {
        "case_id": case_id,
        "decision": decision,
        "risk_score": score,
        "confidence": confidence,
        "signals": signals
    }
```
