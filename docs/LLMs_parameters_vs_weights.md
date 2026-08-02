# LLM Parameters, Weights & Architecture — A Technical Guide

> Understanding what "parameters" means, why weight counts vary, and how model architectures differ across the frontier.

## Definition: Parameters vs. Weights

**Parameter** = a trainable variable — a storage location in the model where a numerical value lives during and after training.

**Weight** = the numerical value stored in that parameter after training has completed.

In practice, engineers and researchers often say "parameter count" and "weight count" interchangeably, because nearly every parameter in a modern neural network stores a learned weight. They're describing the same thing from slightly different angles.

### In code

```python
# PyTorch example
layer = nn.Linear(in_features=10, out_features=5)
# This creates 10×5=50 weight parameters + 5 bias parameters = 55 parameters
# After training, each parameter holds a numerical weight

print(f"Total parameters: {sum(p.numel() for p in layer.parameters())}")
# Output: Total parameters: 55
```

## Intuitive analogy: The neural network as a machine

```
┌────────────────────────────────────────┐
│         Neural Network                 │
├────────────────────────────────────────┤
│ Parameter #1 → Weight = 0.823          │
│ Parameter #2 → Weight = -1.204         │
│ Parameter #3 → Weight = 0.011          │
│ Parameter #4 → Weight = 2.567          │
│        ...                             │
│ Parameter #N → Weight = 0.342          │
└────────────────────────────────────────┘
```

Each parameter is like a dial or lever. Before training, it holds a random or zeroed value. During training, it gets adjusted (via backpropagation) until the model learns something useful. After training, that final adjusted number is the "weight."

## Mathematical view

The core equation of a neural network layer:

```
y = Wx + b
```

Where:
- **x** = input vector
- **W** = weight matrix (each element is a parameter)
- **b** = bias vector (each element is also a parameter)
- **y** = output vector

**Total parameter count** = number of elements in W + number of elements in b.

### Example calculation

```
Input dimension:   10
Output dimension:   5

Weight matrix W:   10 × 5 = 50 parameters
Bias vector b:            =  5 parameters
─────────────────────────────────────────
Total layer:              = 55 parameters
```

Multiply this by the number of layers, and you get the full model size.

## Model size comparisons: Frontier LLMs

| Model family | Disclosed parameter sizes | Transparency & Notes |
|:---|:---|:---|
| **GPT (OpenAI)** | Not disclosed | Exact counts are proprietary. GPT-4 is known to be in the hundreds of billions; exact figure withheld. |
| **Claude (Anthropic)** | Not disclosed | Anthropic does not publish parameter counts for any Claude model. |
| **DeepSeek** | 7B, 67B, 671B (V3) | Open-weight models with published sizes. V3 uses sparse Mixture-of-Experts. |
| **Meta Llama** | 1B, 3B, 8B, 70B, 405B | Dense models. Llama 3.1 405B (Aug 2024) is the largest dense open-weight model. |
| **Grok (xAI)** | Not disclosed | xAI has not published parameter counts. |
| **Perplexity** | N/A | Routing layer over multiple foundation models, not a single proprietary base model. |

### Note on billions (B)

- **1B** = 1 billion parameters (1,000,000,000)
- **7B** = 7 billion parameters
- **70B** = 70 billion parameters
- **405B** = 405 billion parameters

A 70B model typically requires ~140 GB of memory to load in float32 format (2 bytes per parameter × 70 billion). This is why large models are usually deployed in lower-precision formats (float16, int8, int4) to reduce memory footprint.

## Architecture: Dense vs. Mixture-of-Experts (MoE)

### Dense model

In a **dense** model, every parameter participates in every forward pass (every token prediction).

```
┌───────────┐
│   Input   │
└─────┬─────┘
      │
      ▼
  ┌───────────┐
  │ Layer 1   │  ← all 128M parameters active
  └─────┬─────┘
      │
      ▼
  ┌───────────┐
  │ Layer 2   │  ← all 128M parameters active
  └─────┬─────┘
      │
      ▼
┌──────────┐
│  Output  │
└──────────┘

Total compute cost per token ∝ total parameter count
```

**Examples:** Llama 3.1 70B, Claude Sonnet 5, GPT-4.

---

### Mixture-of-Experts (MoE) model

In an **MoE** model, the input is routed to only a subset of "expert" modules. Most parameters remain inactive for any given token.

```
         ┌──────────┐
         │  Input   │
         └────┬─────┘
              │
         ┌────▼────────┐
         │   Router    │  ← which experts to use?
         └────┬────────┘
              │
         ┌────┴────────────────┐
         │                     │
         ▼                     ▼
    ┌────────┐            ┌────────┐
    │Expert 1│            │Expert 2│  ← only 2 of 16 active
    └────┬───┘            └────┬───┘
         │                     │
         │  ┌──────────┐   ┌───┘
         └─►│  Merge   │◄──┘
            └──────────┘
                 │
                 ▼
            ┌────────────┐
            │   Output   │
            └────────────┘

Total compute cost per token ∝ (active parameters × token)
```

**Key insight:** The model may have billions of total parameters, but only a fraction are "active" (participate) in any single prediction. This reduces latency and memory pressure per token.

**Examples:**
- DeepSeek V3: ~671B total parameters, but ~37B active per token (sparse MoE)
- Grok-1: Estimated 314B total, ~85B active (sparse MoE)

---

## When parameter count matters (and when it doesn't)

### More parameters generally → greater capacity

A larger parameter count means the model has more "knobs to turn" during training. It can represent more complex functions and store more knowledge.

### But it's not everything

| Factor | Importance | Example |
|:---|:---|:---|
| **Architecture** | Critical | MoE can be more efficient than dense; Transformer attention mechanisms matter. |
| **Training data quality** | Critical | A 70B model trained on curated data beats a 405B model trained on noisy data. |
| **Training time & compute** | Critical | More steps → better convergence (until diminishing returns). |
| **Parameter count** | Important but not dominant | Capacity, but not destiny. |

### Parameter efficiency

A well-trained smaller model can outperform a poorly-trained large model. And modern sparse architectures (MoE) show that you don't need all parameters active for every prediction.

## Key takeaways

✓ **Parameters define potential capacity** — a 70B model has more "space to learn" than a 7B model.

✓ **Weights are the learned numerical values** — parameters are storage; weights are the data stored after training.

✓ **Larger isn't always better** — architecture, data quality, and training matter as much as raw size.

✓ **Modern frontier models use sparse MoE** — only a subset of parameters are active per token, improving efficiency without sacrificing capability.

✓ **Proprietary models keep counts secret** — OpenAI (GPT), Anthropic (Claude), and xAI (Grok) do not publish exact parameter counts. Open-weight models (Llama, DeepSeek) are transparent.

✓ **Parameter count ≠ capability** — It's one signal among many. Context window, training process, safety tuning, and deployment efficiency all contribute to real-world performance.

## Further reading

- **Transformer paper:** Vaswani et al., "Attention Is All You Need" (2017)  
  https://arxiv.org/abs/1706.03762

- **MoE overview:** Lewis et al., "Base Layers Generalize to Unseen Domains" / Shazeer et al., "Outrageously Large Neural Networks" (2016+)  
  https://arxiv.org/abs/1701.06538

- **Llama 3.1 technical report:** Meta AI (2024)  
  https://llama.meta.com/research

- **DeepSeek V3 technical report:** DeepSeek (2024)  
  https://github.com/deepseek-ai/DeepSeek-V3

---

**Last updated:** August 2026  
**Audience:** Engineers, product managers, and researchers evaluating frontier LLMs
