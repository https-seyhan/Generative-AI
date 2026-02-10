actual calibration exercise.
This is the step almost every AML system skips, and it is exactly why false positives explode.

We will compute thresholds numerically, not heuristically.

Your 3-class decision system:

Class	Meaning
Escalate	Likely financial crime → investigation/SAR
Unclear	Human analyst review
Decline	Benign customer behavior

Your model outputs a risk score (0–1) derived from ML + LLM behavioural signals like:

structuring_score
mule_account_probability
circular_transactions
velocity_anomaly


We now convert that score into decisions.

Step 1 — First Understand the Real Problem

AML is not a classification problem.

It is a loss minimization problem.

You are minimizing:

Operational cost + regulatory exposure + reputational damage

So we assign costs.

Typical (realistic) AML cost matrix
Event	Cost
Investigating innocent customer (false positive)	$45
Missing real laundering case (false negative)	$25,000
Human review case	$12
True detection	$0 (desired outcome)

Notice:
Missing a criminal is ~ 550× worse than annoying a customer.

This single fact determines your thresholds.

Step 2 — Build Validation Data

You need a labelled dataset (very important: not training set)

Example:

Cases	Count
Real laundering	1,200
Benign	98,800
Total	100,000

This imbalance is normal in financial crime.

Step 3 — Your Model Produces a Risk Score

For each customer:

risk_score ∈ [0,1]


Example distribution:

Score Range	Typical Meaning
0.0–0.3	Normal behaviour
0.3–0.6	Suspicious pattern
0.6–1.0	Strong laundering indicators

We now test thresholds.

Step 4 — Try Different Thresholds

We test two cutoffs:

T1 = Decline vs Review
T2 = Review vs Escalate


Example candidates:

T1 = 0.35
T2 = 0.72


Now we simulate outcomes on validation data.

Results at These Thresholds
Model Confusion Outcomes
Decision	Count
Escalate	1,050 cases
Review	4,800 cases
Decline	94,150 cases

Now evaluate correctness:

Outcome	Count
True laundering caught	1,020
Laundering missed	180
Innocent escalated	800
Innocent reviewed	4,200
Step 5 — Compute Financial Loss
1) False Negatives (worst)

Missed criminals:

180 × $25,000 = $4,500,000

2) False Positives

Escalated innocent customers:

800 × $45 = $36,000

3) Human Reviews
4,800 × $12 = $57,600

Total Expected Loss
4,500,000 + 36,000 + 57,600
= $4,593,600

Step 6 — Try Another Threshold (More Conservative)

Now tighten escalation:

T1 = 0.40
T2 = 0.82


New results:

Outcome	Count
True laundering caught	930
Laundering missed	270
Innocent escalated	200
Innocent reviewed	3,000
Recalculate

False negatives:

270 × 25,000 = $6,750,000


False positives:

200 × 45 = $9,000


Reviews:

3,000 × 12 = $36,000

Total:
$6,795,000


This is worse even though alerts dropped.

This is the critical insight:

Reducing alerts does NOT necessarily reduce risk.

You just missed more criminals.

Step 7 — Optimal Threshold

You repeat this over many thresholds and select:

Thresholds that MINIMIZE expected loss


That becomes your regulator-defensible threshold.

Not “0.8 looks good”.
Not “analysts are busy”.

But:

mathematically justified economic decisioning.

How Your Behavioural Signals Create the Score

Your features:

Signal	What it Measures
structuring_score	smurfing deposits
mule_account_probability	account takeover / recruited account
circular_transactions	layering networks
velocity_anomaly	burst movement of funds

Your ML model (e.g., XGBoost or logistic regression) learns:

risk_score =
0.35*structuring +
0.30*mule +
0.20*circular +
0.15*velocity


The LLM contributes by converting narrative patterns:

"multiple small deposits then rapid outgoing transfer"

into quantifiable behavioural indicators.

So the LLM is not the classifier.

It is a behavioural feature extractor.

The ML model is the classifier.

Final Concept (Most Important)

You now have:

Behavioural signals (created by rules + graph + LLM interpretation)

ML model combines them → risk score

Threshold optimization converts score → decision

And this is the crucial shift:

False positives are not fixed by improving the model.
They are fixed by optimizing decision thresholds under cost constraints.
