
# Parameters vs Weights in Neural Networks

## Overview

In deep learning, the terms **parameters** and **weights** are often used interchangeably, but they represent slightly different concepts.

- **Parameter** = a trainable variable that the model learns during training.
- **Weight** = the numerical value stored inside a parameter after training.

**In practice, when people discuss **model size** (for example, a 70B parameter model), they are referring to the total number of trainable parameters.**

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
## Parameter Learning During Training

A neural network learns by updating parameter values through the training process.

### Before Training

┌─────────────────────────┐
│ Parameter #1            │
│                         │
│ Initial Value = 0.215   │
└─────────────────────────┘


⬇️


          Training Process

    ┌───────────────────┐
    │ Forward Pass      │
    │ Loss Calculation  │
    │ Backpropagation   │
    │ Gradient Update   │
    └───────────────────┘


⬇️

### After Training

```
┌─────────────────────────┐
│ Parameter #1            │
│                         │
│ Learned Value = 0.823   │
└─────────────────────────┘
```


## 🔄 What Changes During Training?

During neural network training, the **parameter itself does not move or change identity**.  
The model updates only the **numerical value stored inside that parameter**.

Think of a parameter as a **memory location that stores a learned value**.

### Parameter Update Example


              Training Process
        (Forward Pass → Loss → Backpropagation)
                         │
                         ▼

┌──────────────────┐              ┌──────────────────┐
│  Parameter #1    │              │  Parameter #1    │
│                  │              │                  │
│  Weight = 0.215  │ ───────────▶ │  Weight = 0.823  │
│                  │              │                  │
└──────────────────┘              └──────────────────┘

   Initial Value                    Learned Value

```
The **parameter location remains unchanged**:

```
Parameter #1
     │
     ├── Before Training → 0.215
     │
     └── After Training  → 0.823
```

**Only the stored numerical value is updated.**

---

# 📈 Parameter Learning Lifecycle

| Stage | Parameter State | What Happens |
|:---:|---|---|
| 🟦 **Initialisation** | `Parameter #1 = 0.215` | Model starts with random values or pre-trained weights |
| 🟨 **Training** | `0.215 → 0.823` | Optimisation algorithms update values using gradients |
| 🟩 **Inference** | `Parameter #1 = 0.823` | Learned value is used to generate predictions |

---

> 💡 **Core Concept**
>
> **Parameter = Storage location**  
> **Weight = Numerical value stored in that location**
>
> Training changes the weight value, not the parameter itself.

---

# 🧮 Mathematical View

A neural network layer performs a weighted transformation:

\[
y = Wx + b
\]

Where:

| Symbol | Name | Role |
|:---:|---|---|
| **W** | Weight Matrix | Contains the trainable connection weights between neurons |
| **b** | Bias Vector | Contains additional trainable adjustment values |
| **x** | Input Vector | Data entering the neural network layer |
| **y** | Output Vector | Computed result after applying weights and bias |

---

# Parameter Count Calculation

The total number of trainable parameters in a neural network layer is:
```
\[
\boxed{
\mathrm{Total\ Parameters}
=
\mathrm{Number\ of\ values\ in}\ W
+
\mathrm{Number\ of\ values\ in}\ b
}
\]
```
Where:

- **W** = Weight matrix containing learnable connection values
- **b** = Bias vector containing learnable offset values
```
### Example

For a layer with:

```
Weight Matrix (W)

2 neurons × 3 inputs

W = 6 parameters


Bias Vector (b)

2 neurons

b = 2 parameters
```

The total trainable parameters are:

\[
\text{Total Parameters} = 6 + 2 = 8
\]

Therefore:

```
Total Trainable Parameters = 8
```

> 💡 Every number stored in the weight matrix and bias vector represents a trainable parameter that the model learns during training.

---

## Example Neural Network Layer

```
Input Layer                    Output Layer

x₁ ─────┐
        │
x₂ ─────┼──────► Neuron 1
        │
x₃ ─────┘


Weight Matrix (W)

              x₁      x₂      x₃

Neuron 1    0.12   -0.45    0.77
Neuron 2    0.91    0.33   -0.21


Bias Vector (b)

Neuron 1 → 0.05
Neuron 2 → 0.12
```

Parameter calculation:

```
Weight Matrix (W)

2 neurons × 3 inputs

= 6 parameters


Bias Vector (b)

2 neurons

= 2 parameters


--------------------------

Total Trainable Parameters = 8
```

---

# 🧠 Scaling to Large Language Models

Large Language Models (LLMs) are built from billions of these learned parameters.

```
                 Input Tokens

                      │

                      ▼

          ┌─────────────────────┐
          │ Transformer Network │
          │                     │
          │  Attention Weights  │
          │  Feed Forward       │
          │  Embeddings         │
          │                     │
          │ Billions of Params  │
          └─────────────────────┘

                      │

                      ▼

              Predicted Token
```

The intelligence of a model emerges from the combination of:

```
        Architecture
             +
        Training Data
             +
        Optimisation
             +
        Parameter Scale
             +
        Inference Strategy
```

## 🧠 From Parameters to Intelligence

A large language model is essentially a massive collection of learned parameters:

```
Input Text
    │
    ▼
┌─────────────────────┐
│ Transformer Network │
│                     │
│ Millions / Billions │
│ of Parameters       │
└─────────────────────┘
    │
    ▼
Predicted Output
```

The model's capability emerges from:

```
Architecture
      +
Training Data
      +
Optimisation
      +
Scale of Parameters
      +
Inference Strategy
```

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
