

## How to code  Investigator → structured risk signals
This is the core engineering problem of the whole architecture.

How does the “LLM investigator” turn messy case information into numeric behavioural signals that an ML model can consume?

**This step is not classification and not prompting for a label.**

It is a **cognitive feature extractor**:

        unstructured narrative + events

        -         behavioural analysis
        
        -         structured schema
        
        -         numeric features

### 1. First Principle (Important)

        Do NOT prompt the LLM like this:

                “Is this suspicious? yes/no”

        That destroys calibration and creates false positives.

        Instead you force the LLM to behave like an analyst writing investigation notes.

**The LLM must output signals, not a decision.**

Define a strict schema.

### 2. The Risk Signal Schema (Contract)

        The LLM must always return the same structure.

        This is your feature contract with ML.

**Example:**
```json
        {
          "structuring_score": 0.0-1.0,
          "mule_account_probability": 0.0-1.0,
          "velocity_anomaly": 0.0-1.0,
          "circular_transactions": 0.0-1.0,
          "justification": "short explanation"
        }
```
### 3. Prompt Engineering (This Is the Critical Part)

The LLM does NOT judge.

it's role: <b>financial investigator.</b>

Investigator prompt you should be:
```writing
You are a financial investigations analyst.

Your task is NOT to decide if the case should be escalated.

Your task is to analyze behavioural patterns and produce structured risk signals.

Evaluate the case using behavioural typologies:

1. Structuring:
   Repeated transactions just below common reporting thresholds.

2. Mule account indicators:
   Account acting as pass-through or controlled by third party.

3. Velocity anomaly:
   Transaction speed inconsistent with normal customer behaviour.

4. Circular or layering behaviour:
   Funds returning to originator or moving through chains.

Return ONLY valid JSON using this schema:

{
"structuring_score": float(0-1),
"mule_account_probability": float(0-1),
"velocity_anomaly": float(0-1),
"circular_transactions": float(0-1),
"justification": "max 80 words"
}
```

This instruction (CoT) is extremely important because:

- It prevents the LLM from making the operational decision.5. Feeding Real Case Data

You convert transactions into a narrative summary.

**Example:**
```python
def build_case_text(row):
    return f"""
Customer executed {row['tx_count']} transactions.

Average amount: ${row['avg_amount']}

Largest transaction: ${row['max_amount']}

Unique counterparties: {row['counterparties']}

Average minutes between transfers: {row['avg_interval']}

Repeated identical amounts: {row['repeated_amounts']}
"""
```

- Forcing reasoning → evidence → structured output.

### 4. Python Implementation (investigator_llm.py)
```python

import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt Template


SYSTEM_PROMPT = """
You are a financial investigations analyst.

Your task is NOT to decide escalation.

Analyze behaviour and output structured risk signals.

Return ONLY JSON:
{
 "structuring_score": float(0-1),
 "mule_account_probability": float(0-1),
 "velocity_anomaly": float(0-1),
 "circular_transactions": float(0-1),
 "justification": "short explanation"
}
"""


# Investigator Function

def investigate_case(case_text):

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case_text}
        ]
    )

    content = response.choices[0].message.content

    # Parse JSON safely
    try:
        signals = json.loads(content)
    except:
        raise ValueError("LLM returned non-JSON output")

    return signals
```
### 5. Feeding Real Case Data

<b>Convert transactions into a narrative summary.</b>

**Example:**

```python
def build_case_text(row):
    return f"""
Customer executed {row['tx_count']} transactions.

Average amount: ${row['avg_amount']}

Largest transaction: ${row['max_amount']}

Unique counterparties: {row['counterparties']}

Average minutes between transfers: {row['avg_interval']}

Repeated identical amounts: {row['repeated_amounts']}
"""
```
**Then:**
```python
signals = investigate_case(build_case_text(row))
```

**The Output of above code:**
```python
{
 "structuring_score": 0.82,
 "mule_account_probability": 0.71,
 "velocity_anomaly": 0.44,
 "circular_transactions": 0.65
}
```
- **$${\color{blue} This \ is \ exactly \ what \ LightGBM \ model \ needs}    $$**

### 6. Converting to ML Features

```python
def to_feature_vector(signals):

    return [
        signals["structuring_score"],
        signals["mule_account_probability"],
        signals["velocity_anomaly"],
        signals["circular_transactions"]
    ]
```

**That vector becomes:**

```python
X = [0.82, 0.71, 0.44, 0.65]
```

**Now ML can calibrate risk.**

### Why This Works

You have separated cognition from statistics.

**LLM:**
```python
pattern recognition + behavioural understanding
```

**ML:**
```python
probability estimation
```

The LLM does something classical feature engineering cannot do:

It **interprets intent-like behaviour** from messy evidence.

The ML model then learns how much each behavioural pattern actually matters in real outcomes.

What You Just Implemented

This investigator is equivalent to a junior analyst writing case notes, except:

consistent

fast

structured

measurable

You did not automate decisions.

You automated evidence generation.

And that is precisely why the hybrid LLM-ML architecture becomes reliable

