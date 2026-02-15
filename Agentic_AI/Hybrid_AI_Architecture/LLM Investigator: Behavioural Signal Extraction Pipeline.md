
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
```json
{
"structuring_score": float(0-1),
"mule_account_probability": float(0-1),
"velocity_anomaly": float(0-1),
"circular_transactions": float(0-1),
"justification": "max 80 words"
}
```
```

This instruction (CoT) is extremely important because:

- It prevents the LLM from making the operational decision.

- Forcing reasoning → evidence → structured output.
