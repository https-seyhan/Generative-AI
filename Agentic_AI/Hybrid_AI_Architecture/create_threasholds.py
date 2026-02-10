self-contained Python script that derives the two operational thresholds automatically from your model outputs.

It does two things simultaneously:

Expected-loss minimisation (risk economics — regulator defensible)

Analyst capacity control (operations — actually workable)

This is exactly how mature AML monitoring systems are tuned.

What you need first

After training your classifier, run it on historical labelled cases (VERY important — not training data, use hold-out or last year’s alerts).

Create a CSV like this:

model_scores.csv

y_true	p_escalate
0	0.02
0	0.11
1	0.37
2	0.84
0	0.09
2	0.92

Where:

y_true:
0 = decline (not suspicious)
1 = unclear (review-worthy)
2 = escalate (SAR-quality confirmed)

p_escalate:
model probability P(laundering)

threshold_optimizer.py
import pandas as pd
import numpy as np

# -----------------------------
# CONFIGURATION (EDIT THESE)
# -----------------------------

C_REVIEW = 80              # analyst investigation cost
C_FALSE_ESCALATE = 800     # unnecessary SAR cost
C_MISSED_CRIME = 80000     # missing real laundering

ANALYST_CAPACITY_PER_DAY = 500
DAYS_IN_SAMPLE = 180       # days covered by your dataset


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("model_scores.csv")

y = df["y_true"].values
p = df["p_escalate"].values

# binary ground truth for laundering
is_crime = (y == 2).astype(int)

# -----------------------------
# EXPECTED LOSS FUNCTION
# -----------------------------
def expected_loss(p, T_review, T_escalate):

    loss = 0

    for prob in p:

        # decline
        if prob < T_review:
            loss += prob * C_MISSED_CRIME

        # review
        elif T_review <= prob < T_escalate:
            loss += C_REVIEW

        # escalate
        else:
            loss += (1 - prob) * C_FALSE_ESCALATE

    return loss / len(p)


# -----------------------------
# CAPACITY CONSTRAINT
# -----------------------------
def alerts_per_day(p, T_review):
    reviews = np.sum(p >= T_review)
    return reviews / DAYS_IN_SAMPLE


# -----------------------------
# SEARCH OPTIMAL THRESHOLDS
# -----------------------------
best_loss = 1e18
best_T_review = None
best_T_escalate = None

review_range = np.arange(0.05, 0.70, 0.01)
escalate_range = np.arange(0.50, 0.99, 0.01)

for T_r in review_range:

    # enforce analyst capacity
    if alerts_per_day(p, T_r) > ANALYST_CAPACITY_PER_DAY:
        continue

    for T_e in escalate_range:
        if T_e <= T_r:
            continue

        loss = expected_loss(p, T_r, T_e)

        if loss < best_loss:
            best_loss = loss
            best_T_review = T_r
            best_T_escalate = T_e


# -----------------------------
# PERFORMANCE REPORT
# -----------------------------
decline = p < best_T_review
review = (p >= best_T_review) & (p < best_T_escalate)
escalate = p >= best_T_escalate

precision = np.sum(is_crime & escalate) / max(np.sum(escalate),1)
recall = np.sum(is_crime & escalate) / max(np.sum(is_crime),1)

print("\nOptimal Thresholds")
print("------------------")
print(f"T_review   : {best_T_review:.2f}")
print(f"T_escalate : {best_T_escalate:.2f}")

print("\nOperational Metrics")
print("------------------")
print(f"Reviews/day      : {alerts_per_day(p, best_T_review):.0f}")
print(f"Escalations/day  : {np.sum(escalate)/DAYS_IN_SAMPLE:.0f}")
print(f"Escalation Precision : {precision:.3f}")
print(f"Crime Recall         : {recall:.3f}")
print(f"Expected Loss        : ${best_loss:,.2f}")

How to run
python threshold_optimizer.py


You will get output like:

Optimal Thresholds
T_review   : 0.43
T_escalate : 0.81

Operational Metrics
Reviews/day      : 492
Escalations/day  : 38
Escalation Precision : 0.72
Crime Recall         : 0.61
Expected Loss        : $21,480

What the script actually did (important)

The program simulated every possible policy:

For each customer probability it asked:

Which action (decline / review / escalate) produces the lowest financial-crime risk cost while keeping analysts within capacity?

This is not ML tuning.

This is decision optimisation — the missing layer in almost all academic AML models.

Why this dramatically reduces false positives

Because false positives come from:

Treating weak evidence as strong evidence.

The optimisation forces:

low-probability cases → decline

ambiguous → analysts

only strong → escalate

So the SAR queue becomes high precision.

What you now have

You now built a complete regulatory-grade pipeline:

LLM → behavioural signals
        ↓
ML classifier → calibrated probability
        ↓
Threshold optimizer → operational policy
        ↓
Decline / Review / Escalate


This final step (threshold optimisation) is actually what transforms a model into a financial crime monitoring system rather than a research classifier.
