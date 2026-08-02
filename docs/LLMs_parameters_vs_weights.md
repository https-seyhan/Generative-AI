
# Parameters vs Weights in Neural Networks

## Overview

In deep learning, the terms **parameters** and **weights** are often used interchangeably, but they represent slightly different concepts.

- **Parameter** = a trainable variable that the model learns during training.
- **Weight** = the numerical value stored inside a parameter after training.

In practice, when people discuss **model size** (for example, a 70B parameter model), they are referring to the total number of trainable parameters.

---

# Parameter vs Weight

| Concept | Definition |
|---|---|
| Parameter | A location in the model that can be adjusted during training |
| Weight | The actual numerical value stored in that parameter |
| Training | Updates parameter values using optimisation algorithms |
| Model Size | Usually measured by total parameter count |

Example:

```
Parameter #1  → stores →  Weight = 0.823

Parameter #2  → stores →  Weight = -1.204

Parameter #3  → stores →  Weight = 0.011
```

The parameters are the **containers**.  
The weights are the **learned values inside those containers**.

---

# Neural Network Analogy

```
┌──────────────────────────────┐
│ Parameter #1 = Weight  0.823 │
│ Parameter #2 = Weight -1.204 │
│ Parameter #3 = Weight  0.011 │
└──────────────────────────────┘
```

During training:

```
Before Training

Parameter #1 = 0.215


        Training Process
              │
              ▼


After Training

Parameter #1 = 0.823
```

The parameter location remains the same, but the stored value changes.

---

# Mathematical View

A simple neural network layer:

\[
y = Wx + b
\]

Where:

| Symbol | Meaning |
|---|---|
| `W` | Weight matrix (trainable parameters) |
| `b` | Bias parameters |
| `x` | Input vector |
| `y` | Output vector |

Total model parameters:

\[
\text{Total Parameters} = \text{All Values in } W + \text{All Values in } b
\]

Example:

```
Weight Matrix (W)

       0.12   -0.45
W =    0.77    0.91
       0.33   -0.21


Bias (b)

b = [0.05, 0.12]
```

All numerical values contribute to the parameter count.

---

# Why Parameter Count Matters

Parameter count is commonly used as a rough indicator of:

- Model capacity
- Memory requirements
- Training compute requirements
- Inference cost
- Potential reasoning capability

However:

> More parameters do not automatically mean a better model.

Performance also depends on:

- Training data quality
- Architecture design
- Optimisation techniques
- Alignment methods
- Fine-tuning strategy

---

# Foundation Model Parameter Comparison

| Model Family | Public Parameter Sizes | Notes |
|---|---|---|
| GPT (OpenAI) | Not disclosed | Exact parameter counts are proprietary |
| Claude (Anthropic) | Not disclosed | Anthropic does not publish model sizes |
| DeepSeek | 7B, 67B, V3 (~671B total MoE) | Open-weight research models; MoE uses sparse activation |
| Meta Llama | 1B, 3B, 8B, 70B, 405B | Dense transformer models depending on release |
| Grok (xAI) | Not disclosed | xAI has not published exact counts |
| Perplexity | Not applicable | Uses multiple underlying foundation models |

---

# Dense Models vs Mixture-of-Experts (MoE)

Modern large language models are increasingly moving from purely dense architectures to **Mixture-of-Experts (MoE)** architectures.

---

# Dense Transformer Model

A dense model activates all parameters for every token prediction.

```
              Input Token

                  │

                  ▼

        ┌─────────────────┐
        │                 │
        │  Transformer    │
        │                 │
        │ ███████████████ │
        │ All Parameters  │
        │ Active          │
        └─────────────────┘

                  │

                  ▼

              Output Token
```

Example:

```
70B Dense Model

Every token uses:

70 Billion parameters
```

---

# Mixture-of-Experts (MoE) Model

An MoE model contains multiple expert networks.

A router decides which experts are activated.

```
                 Input Token

                      │

                      ▼

                  ┌────────┐
                  │ Router │
                  └────────┘

          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼

      Expert 1  Expert 2  Expert 3

          │        │
          │        │
          ▼        ▼

      Selected Experts Only


                 │

                 ▼

             Output Token
```

Example:

```
671B Total Parameters

but only:

~37B parameters active per token
```

This provides:

- Larger total model capacity
- Lower inference cost
- Better scaling efficiency

---

# Dense vs MoE Comparison

| Feature | Dense Model | MoE Model |
|---|---|---|
| Parameter usage | All parameters active | Subset activated |
| Memory footprint | Higher | Lower active memory |
| Compute cost | Higher | More efficient |
| Architecture | Single network | Multiple expert networks |
| Routing | Not required | Router selects experts |
| Scaling | Expensive | More scalable |

---

# Parameter Memory Calculation

Model memory depends on:

\[
\text{Memory} =
\text{Number of Parameters}
\times
\text{Bytes per Parameter}
\]

Examples:

## FP32

```
1 parameter = 4 bytes
```

A 70B parameter model:

```
70,000,000,000 × 4 bytes

≈ 280 GB memory
```

---

## FP16

```
1 parameter = 2 bytes
```

70B model:

```
70,000,000,000 × 2 bytes

≈ 140 GB memory
```

---

## INT8 Quantisation

```
1 parameter = 1 byte
```

70B model:

```
≈ 70 GB memory
```

---

# Key Takeaways

✅ **Parameters define model capacity.**

✅ **Weights are the learned numerical values stored inside parameters.**

✅ **Parameter count is the number of trainable values in a model.**

✅ **Larger parameter counts generally increase model capacity, but architecture and data quality are equally important.**

✅ **Modern frontier models increasingly use sparse Mixture-of-Experts architectures where only a subset of parameters participate in each prediction.**

---

# Summary

```
Parameter
    │
    ▼
Trainable location in the model
    │
    ▼
Stores
    │
    ▼
Weight value after training
    │
    ▼
Used by neural network computation
```

A model's intelligence is not determined only by the number of parameters, but by the combination of:

```
Model Architecture
        +
Training Data
        +
Optimisation
        +
Parameter Scale
        +
Inference Strategy
```
