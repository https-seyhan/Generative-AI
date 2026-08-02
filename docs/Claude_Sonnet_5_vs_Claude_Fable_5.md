# Claude Sonnet 5 vs Claude Fable 5

I'm Sonnet 5, so this is partly me sizing myself up against the
flagship. Quick framing: these aren't really peers --- they sit at
different rungs of the current Anthropic lineup. Fable 5 is Anthropic's
most capable widely released model, built for the most demanding
reasoning and long-horizon agentic work --- the first "Mythos-class"
model, positioned above Opus. Sonnet 5 sits in the middle of the stack,
between Haiku and Opus, built as the everyday agentic/coding workhorse.

  -------------------------------------------------------------------------
  Feature                 Claude Sonnet 5         Claude Fable 5
  ----------------------- ----------------------- -------------------------
  Tier                    Mid-tier ("Sonnet")     Flagship ("Mythos-class")

  Released (GA)           June 30, 2026           June 9, 2026

  Pricing (in / out per   \$3 / \$15 standard;    \$10 / \$50
  MTok)                   \$2 / \$10 introductory 
                          through Aug 31, 2026    

  Context window          1M tokens (fixed --- no 1M tokens
                          smaller option)         

  Max output              128K tokens             128K tokens

  Adaptive thinking       On by default, can be   Always on, cannot be
                          disabled                disabled

  Manual/budgeted         Not supported (400      Not supported
  thinking                error)                  

  Comparative latency     Fast                    Slower

  Reliable knowledge      Jan 2026                Jan 2026
  cutoff                                          

  Safety classifiers      Cybersecurity only      Cybersecurity +
                                                  biology/life-sciences +
                                                  reasoning-extraction

  Zero data retention     Supported for eligible  Not eligible (30-day
                          orgs                    minimum retention)
  -------------------------------------------------------------------------

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

