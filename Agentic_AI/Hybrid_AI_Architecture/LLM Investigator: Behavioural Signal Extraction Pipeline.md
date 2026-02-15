
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

        Example:
