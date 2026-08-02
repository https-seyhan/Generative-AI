# Claude Sonnet 5 vs Claude Fable 5

Quick framing: these aren't really peers --- they sit at
different rungs of the current Anthropic lineup. Fable 5 is Anthropic's
most capable widely released model, built for the most demanding
reasoning and long-horizon agentic work --- the first "Mythos-class"
model, positioned above Opus. Sonnet 5 sits in the middle of the stack,
between Haiku and Opus, built as the everyday agentic/coding workhorse.

# 📊 Claude Sonnet 5 vs Claude Fable 5

| Feature | 🟦 Claude Sonnet 5 | 🟥 Claude Fable 5 |
|:--------|:-------------------|:------------------|
| **Tier** | Mid-tier (**Sonnet**) | Flagship (**Mythos-class**) |
| **Released (GA)** | 30 June 2026 | 9 June 2026 |
| **Pricing**<br>*(Input / Output per million tokens)* | **$3 / $15** (Standard)<br>**$2 / $10** introductory until **31 Aug 2026** | **$10 / $50** |
| **Context Window** | **1M tokens** (fixed – no smaller option) | **1M tokens** |
| **Maximum Output** | **128K tokens** | **128K tokens** |
| **Adaptive Thinking** | Enabled by default (can be disabled) | Always enabled (cannot be disabled) |
| **Manual/Budgeted Thinking** | ❌ Not supported (400 error) | ❌ Not supported |
| **Latency** | ⚡ Faster | 🐢 Slower |
| **Knowledge Cut-off** | January 2026 | January 2026 |
| **Safety Classifiers** | 🛡️ Cybersecurity | 🛡️ Cybersecurity<br>🧬 Biology / Life Sciences<br>🧠 Reasoning Extraction |
| **Zero Data Retention (ZDR)** | ✅ Supported for eligible organisations | ❌ Not eligible (minimum 30-day retention) |

## Key Takeaways

### Price

Fable runs roughly **3.3×** Sonnet's standard cost (or **5×** the
introductory rate).

### Capability

Fable targets the hardest reasoning and long-running agentic tasks.
Sonnet delivers excellent performance for everyday coding, reasoning,
and agentic workloads.

### Safety

Both models can refuse requests as successful API responses rather than
errors. Sonnet focuses on cybersecurity safeguards, while Fable
additionally covers biology/life sciences and reasoning extraction.

### Data Retention

-   **Sonnet 5:** Supports Zero Data Retention (eligible organisations).
-   **Fable 5:** Mandatory 30-day retention.

### Tokenizer

Both models use the newer tokenizer producing approximately **30% more
tokens** than the previous generation.

### Launch History

Fable experienced a temporary suspension due to US export-control
directives before being restored. Sonnet launched without interruption.

## Recommendation

  Use Case                         Recommended Model
  -------------------------------- -------------------
  High-volume coding               Sonnet 5
  Cost-sensitive agentic systems   Sonnet 5
  Long-horizon reasoning           Fable 5
  Highest-stakes reasoning         Fable 5

