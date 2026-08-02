# Claude Sonnet 5 vs Claude Fable 5 — Comparison Guide (Australia Edition)

**Last updated:** 2 August 2026
**Prepared for:** Seyhan — real estate agentic AI / RAG platform (Australian market)

> **Disclaimer:** This is general technical and pricing information, not legal, tax, or financial advice. Currency conversions are approximate and move daily. Confirm GST treatment with your accountant and in your Claude Console billing settings, and confirm privacy obligations with a qualified Australian privacy lawyer before relying on this for compliance or budgeting decisions.

## Contents

- [At a glance](#at-a-glance)
- [Pricing in AUD](#pricing-in-aud)
- [Capability and behaviour](#capability-and-behaviour)
- [Safety classifiers and refusals](#safety-classifiers-and-refusals)
- [Data retention](#data-retention)
- [Availability and recent history](#availability-and-recent-history)
- [Australian regulatory considerations](#australian-regulatory-considerations)
- [When to use which](#when-to-use-which)
- [Sources](#sources)

## At a glance

Fable 5 and Sonnet 5 sit at different rungs of Anthropic's current line-up, not side by side. Fable 5 is the flagship — Anthropic's first "Mythos-class" model, positioned above Opus, built for the hardest reasoning and longest-running agentic work. Sonnet 5 is the mid-tier workhorse, sitting between Haiku and Opus, tuned for everyday agentic and coding work at volume.

### Core specifications

| Attribute | **Sonnet 5** | **Fable 5** |
|:---|:---|:---|
| **Tier** | Mid-tier | Flagship (Mythos-class) |
| **Model ID** | `claude-sonnet-5` | `claude-fable-5` |
| **Release date** | 30 June 2026 | 9 June 2026 |

### Capacity and performance

| Attribute | **Sonnet 5** | **Fable 5** |
|:---|:---|:---|
| **Context window** | 1M tokens (fixed) | 1M tokens |
| **Max output** | 128K tokens | 128K tokens |
| **Adaptive thinking** | ✓ On by default, can disable | ✓ Always on, tunable only |
| **Manual thinking** | ✗ Not supported | ✗ Not supported |
| **Typical latency** | Fast | Slower (reasoning overhead) |
| **Knowledge cutoff** | January 2026 | January 2026 |

### Safety & compliance

| Attribute | **Sonnet 5** | **Fable 5** |
|:---|:---|:---|
| **Safety classifiers** | Cybersecurity | Cybersecurity + biology/life-sciences + reasoning extraction |
| **Refusal handling** | HTTP 200 with `stop_reason: "refusal"` | HTTP 200 with `stop_reason: "refusal"` |
| **Fallback supported** | Yes — server, client, or manual | Yes — server, client, or manual |
| **Zero data retention** | ✓ Supported (with agreement) | ✗ 30-day minimum (no ZDR option) |

## Pricing in AUD

Anthropic bills the Claude API in USD — there's no separate published AUD price list. Figures below are converted at approximately **1 USD ≈ 1.42 AUD** (indicative mid-market rate, early August 2026). Check a live rate before budgeting, since it moves daily.

#### Sonnet 5 pricing

| Period | Input | Output |
|:---|:---:|:---:|
| **USD** (introductory, through 31 Aug 2026) | $2 / MTok | $10 / MTok |
| **AUD** (ex-GST) | ~A$2.84 | ~A$14.20 |
| **AUD** (incl. 10% GST) | ~A$3.12 | ~A$15.62 |
| | | |
| **USD** (standard, from 1 Sept 2026) | $3 / MTok | $15 / MTok |
| **AUD** (ex-GST) | ~A$4.26 | ~A$21.30 |
| **AUD** (incl. 10% GST) | ~A$4.69 | ~A$23.43 |

#### Fable 5 pricing

| Period | Input | Output |
|:---|:---:|:---:|
| **USD** | $10 / MTok | $50 / MTok |
| **AUD** (ex-GST) | ~A$14.20 | ~A$71.00 |
| **AUD** (incl. 10% GST) | ~A$15.62 | ~A$78.10 |

> **Per million tokens (MTok).** GST treatment assumes default consumer-style charge; ABN-registered businesses should confirm reverse-charge mechanics with their accountant and in Console settings.

**On the 10% GST:** Anthropic bills Australian customers under a non-resident digital-services registration and applies GST by default. If your business is GST-registered, adding your ABN under **Settings → Organization** in the Claude Console is typically what removes GST from future invoices, shifting it to a reverse charge you account for yourself in your BAS. Treat that as a starting point for a conversation with your accountant rather than a settled position — the exact mechanics depend on how the supply is characterised. See Anthropic's guide on [adding a tax/VAT ID](https://support.claude.com/en/articles/9889428-add-or-update-your-claude-console-organization-s-tax-or-vat-id).

**Cost gap:** Fable 5 runs roughly **3.3x** Sonnet 5's cost at Sonnet's standard rate, or about **5x** at Sonnet's current introductory rate — consistent across both input and output tokens.

## Capability and behaviour

Anthropic's own Sonnet 5 system card is fairly candid about the gap: Sonnet 5 posts clear gains over its predecessor on coding, agentic search, and reasoning, but trails the Opus and Mythos-class models (Fable's tier) on most evaluations. Fable's premium is aimed at the hardest, longest-running tasks rather than a blanket quality upgrade for everyday work.

Thinking behaviour differs in a way that affects cost predictability. Sonnet 5 has adaptive thinking on by default but it can be switched off (`thinking: {type: "disabled"}`). Fable 5's thinking is always on and can only be tuned down via the `effort` parameter, not disabled outright. Neither model supports the old manual/budgeted thinking mode — that returns a 400 error on both, same as on Opus 4.8 and 4.7.

Both models moved to Anthropic's newer tokenizer, which produces roughly 30% more tokens than the previous generation for the same text (Fable 5 vs. pre-Opus-4.7 models; Sonnet 5 vs. Sonnet 4.6). Re-measure your RAG prompts before setting token budgets rather than reusing old counts — for cost forecasting, this matters more than the headline per-token price.

## Safety classifiers and refusals

Both models can now decline a request as a normal response rather than an error. The Messages API returns `stop_reason: "refusal"` as a successful HTTP 200, reports which classifier triggered it, and you aren't billed for the refused request. You can configure server-side, client-side, or manual fallback to retry on another model, with fallback credit refunding the prompt-cache cost of switching.

The scope of what gets screened differs. Sonnet 5 is the first Sonnet-tier model with real-time cybersecurity safeguards, and that's the only domain it screens. Fable 5's classifiers additionally cover biology/life-sciences content and attempts to extract its own reasoning. Fable's sibling, Mythos 5, is the same underlying model with those classifiers removed, but it's invite-only through Project Glasswing — Fable is the version everyone else uses.

For an agentic RAG pipeline processing retrieved content from third-party sources (listings, contracts, correspondence), the wider refusal surface on Fable is worth factoring into fallback design for anything routed to that tier.

## Data retention

This is the detail most worth flagging for a platform handling buyer, seller, and landlord information:

- **Fable 5 and Mythos 5** carry a mandatory 30-day retention window and are **not available under zero data retention** — Anthropic designates them "Covered Models" for retention purposes.
- **Sonnet 5 supports zero data retention** for organisations with a ZDR agreement in place.

This interacts directly with Privacy Act obligations — see below.

## Availability and recent history

Fable 5 had a rocky start: Anthropic suspended it worldwide on 12 June 2026 to comply with a US export-control directive, then restored access on 1 July 2026 after the Commerce Department narrowed the rule (see Anthropic's [statement](https://www.anthropic.com/news/redeploying-fable-5)). Sonnet 5 launched three weeks later, on 30 June 2026, with no such interruption. Both models are available through the Claude API, Amazon Bedrock, Claude Platform on AWS, Google Cloud, and Microsoft Foundry.

## Australian regulatory considerations

None of this changes because you picked Sonnet over Fable or vice versa — both models are served from Anthropic's infrastructure unless you specifically deploy through a region-locked path. The obligations below apply to either model; the retention difference above just shifts your risk profile slightly.

### Privacy Act 1988 (Cth) and the Australian Privacy Principles

A platform handling real estate buyer, seller, and landlord personal information is an "APP entity" under the *Privacy Act 1988* (Cth), governed by the 13 Australian Privacy Principles (APPs). Three matter most for an AI/RAG pipeline:

- **APP 8 — Cross-border disclosure.** Before personal information reaches an overseas recipient — including an AI model processing it on servers outside Australia — you must take reasonable steps to ensure that recipient handles it consistently with the APPs. Under section 16C, if the overseas recipient (e.g. Anthropic) mishandles the data, **you're accountable as if you'd mishandled it yourself.** This applies whether the disclosure is deliberate or just incidental to how your RAG pipeline calls the API.
- **APP 11 — Security of personal information.** You must take reasonable steps to protect personal information from misuse, loss, and unauthorised access, and to destroy or de-identify it once it's no longer needed. The underlying model's retention practice is directly relevant here — a 30-day mandatory floor (Fable/Mythos) versus zero data retention (Sonnet 5, with an agreement) changes what "reasonable steps" looks like for that portion of your pipeline.
- **APP 6 — Use and disclosure.** Personal information collected for one purpose (matching a buyer to a listing, say) generally can't be used for another purpose without consent — worth checking against any plan to use retained conversation data for model evaluation or improvement.

The Privacy Act has been under active reform (the *Privacy and Other Legislation Amendment Act 2024* and subsequent tranches), including new transparency obligations around automated decision-making that could bite if your agents make or materially influence decisions about buyers or applicants. Treat the current Act as a floor, not the ceiling, and check for amendments in force as at your review date.

### Practical steps

- Run a Privacy Impact Assessment (PIA) specifically for the agentic/RAG use case, not just for the platform generally.
- Check Anthropic's Data Processing Addendum and sub-processor list, and reflect the overseas disclosure (United States) in your privacy policy and collection notices.
- Consider **AWS Bedrock in the Sydney region (`ap-southeast-2`)**, using AU-scoped cross-region inference profiles, if you want inference to stay within Australia rather than relying on Anthropic's global API — this simplifies the APP 8 analysis considerably. Confirm current-generation model availability in-region before architecting around it, as regional rollout can lag global release.
- For the highest-sensitivity data (financial capacity, ID documents), lean toward whichever configuration gives you zero data retention, or de-identify before it ever reaches the model.

## When to use which

### Claude Sonnet 5 ✓

**Best for:**
- High-volume, cost-sensitive agentic and coding work
- Anything where zero data retention (ZDR) matters for compliance
- The sensible default for most of a production RAG pipeline
- Real-time processing with tight latency requirements

**Why:** Fastest option, lowest cost, supports ZDR agreements, cybersecurity classifiers sufficient for most use cases.

---

### Claude Fable 5 ⚡

**Best for:**
- Hardest, highest-stakes reasoning tasks
- Long-horizon multi-step agentic workflows
- Work that justifies a 3–5x cost multiplier through superior capability
- Scenarios where reasoning-extraction classifier matters (unusual — most real estate workflows don't)

**Why:** Frontier capabilities, always-on adaptive thinking, broader safety classification.

**Trade-offs:** Slower, more expensive, 30-day mandatory retention (no ZDR), wider refusal surface requiring fallback handling.

---

**Recommendation for your real estate platform:** Route most conversation and RAG ingestion through Sonnet 5, treating Fable 5 as a high-stakes fallback for complex multi-agent reasoning tasks (e.g. cross-document legal analysis). Wire up conditional routing and monitor fallback rates in the early rollout phase.

## Sources

**Official (Anthropic / Australian Government):**

- Anthropic, [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- Anthropic, [Introducing Claude Fable 5 and Claude Mythos 5](https://platform.claude.com/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5)
- Anthropic, [What's new in Claude Sonnet 5](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5)
- Anthropic, [Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5)
- Anthropic, [Claude Sonnet 5 System Card](https://www-cdn.anthropic.com/480e0bb54327b9622282e9c39a83a4f490ed377e/Claude%20Sonnet%205%20System%20Card.pdf) (30 June 2026)
- Anthropic, [statement on restoring Fable 5 access](https://www.anthropic.com/news/redeploying-fable-5)
- Anthropic, [Billing address and tax calculation](https://support.claude.com/en/articles/12997130-understanding-your-billing-address-and-tax-calculation); [Console tax/VAT ID](https://support.claude.com/en/articles/9889428-add-or-update-your-claude-console-organization-s-tax-or-vat-id)
- Office of the Australian Information Commissioner (OAIC), [APP Guidelines — Chapter 8: Cross-border disclosure of personal information](https://www.oaic.gov.au/__data/assets/pdf_file/0036/256959/APP-Guidelines-Chapter-8-Cross-border-disclosure-of-personal-information-October-2025-v1.3.PDF)

**Industry/practitioner commentary (useful context — verify specifics before relying on for compliance or tax decisions):**

- Mondaq, [Australian Privacy Compliance: Four Key Developments in 2026](https://www.mondaq.com/australia/privacy-protection/1798794/)
- PrivacyReady, [APP 8 Explained: Cross-Border Disclosure](https://privacyready.com.au/learn/app-8-cross-border-disclosure)
- PADISO, [Deploying Claude in Australia: Data Residency, Compliance, and Latency](https://www.padiso.co/blog/deploying-claude-australia-data-residency-compliance-latency/)
- Sandlabs, [Claude AI Pricing 2026](https://sandlabs.com.au/blog/claude-ai-pricing-guide) (GST/ABN treatment)
- XE, live [USD/AUD exchange rate](https://www.xe.com/en-us/currencyconverter/convert/?Amount=1&From=USD&To=AUD)

