<b> Behavioural signals are not invented by the LLM and not decided by a data scientist alone.</b>

<b>In a real financial-crime system they come from regulatory typologies.</b>


**So the origin chain is:**

	Global crime patterns
	&nbsp; 
    ↓
	&nbsp; 
	Regulators & FIUs publish typologies
    ↓
	AML investigators interpret them
    ↓
	Bank risk team formalises indicators
    ↓
	Engineers convert to measurable signals
    ↓
	LLM helps recognise them in narratives
    ↓
	ML model uses them for probability estimation


Let’s go layer by layer.

1) Who defines the behaviours (outside the bank)

Financial-crime behaviour patterns are documented internationally.
They are called typologies.

They come from:

• Financial Intelligence Units (FIUs)
• Central banks / regulators
• Law enforcement investigations
• Court cases

Examples (Australia context):

AUSTRAC case reports

FATF (Financial Action Task Force) typology papers

Egmont Group intelligence bulletins

These bodies study real criminal investigations and publish patterns such as:

“Launderers use multiple small deposits across locations followed by transfers to newly created accounts.”

That sentence is literally the origin of your structuring_score.

So the signals are evidence-based, not ML-invented.

2) Who defines them inside a bank

Inside a financial institution, three groups collaborate:

(a) AML Investigators (most important)

They know how real suspicious matters look.

They answer:

what actually led to SAR filings

which alerts were false positives

what patterns repeat

They are the subject matter experts.

(b) Financial Crime Compliance (FCC) / Risk Policy

They translate investigator experience into a policy definition:

Example:

“Repeated deposits below reporting threshold across multiple branches within short time constitutes structuring risk.”

This becomes a risk indicator definition.

(c) Data Science + Financial Crime Analytics

They operationalise the policy into measurable features.

This is where the signal is mathematically constructed.

3) How a behavioural signal is actually engineered

Take one example: structuring.

Step 1 — Legal / investigative observation

Investigators observe cases:

Criminals avoid cash reporting thresholds.

Step 2 — Policy statement

Compliance defines a rule:

Activity near reporting threshold is suspicious when repeated and geographically dispersed.

Step 3 — Quantifiable variables

Engineers create measurable components:

Observable	Data field
near threshold	transaction amount
repetition	frequency
dispersion	branch/ATM location
timing	timestamps
Step 4 — Mathematical feature

Example engineered metric:

structuring_score=w1⋅amount_proximity+w2⋅frequency+w3⋅location_dispersion+w4⋅temporal_pattern
structuring_score=w
1
	​

⋅amount_proximity+w
2
	​

⋅frequency+w
3
	​

⋅location_dispersion+w
4
	​

⋅temporal_pattern

Now it becomes a machine-readable signal.

Step 5 — LLM role

The LLM does not calculate the metric.

Instead it reads a summarised transaction history like:

14 deposits between $8,900–$9,900 across 5 ATMs in 3 days

and recognises the behavioural concept:

intentional avoidance

It outputs the score.

So:

Component	Responsibility
Definition	investigators/regulators
Measurement	engineers/data science
Recognition	LLM
Decision	calibrated ML classifier
Action	compliance operations
4) Why LLMs became useful here

Traditional rule systems required thousands of handcrafted rules.

Problem:
Criminal behaviour adapts faster than rule writing.

LLMs help because they detect semantic behaviour, not only numeric thresholds.

They approximate how an investigator reads a case narrative.

They convert:

transaction history → behavioural description


Then ML converts:

behavioural description → calibrated probability

5) Important governance point

For regulators, behavioural signals must be:

• explainable
• documented
• traceable to typologies
• independently reviewable

You should maintain a typology register, for example:

Signal	Source	Justification
structuring_score	FATF placement typology	threshold avoidance
mule_account_probability	law enforcement cases	pass-through accounts
circular_transactions	layering typology	origin obfuscation
velocity_anomaly	rapid movement typology	trace disruption

This is what model risk management and auditors check.

Key takeaway

Behavioural risk signals are not a machine-learning invention.

They are:

codified investigator knowledge about how money laundering actually happens, translated into quantitative features so a model can reason consistently and at scale.

The LLM’s job is simply to recognise the pattern humans already understand, while the ML model ensures consistent, calibrated decisions across millions of customers.
