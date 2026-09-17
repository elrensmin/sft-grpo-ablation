# SFT-GRPO ablation study

## Question

When does an SFT stage before GRPO produce a net positive effect on both target-task performance and general capability retention and how do we configure the SFT stage (LoRA rank, target modules, data quantity) to maximize that net effect?

## Hypothesis

We expect SFT → GRPO (P2) to beat GRPO alone (P0) on GSM8K while retaining general capability better, i.e. SFT provides a warm start that RL then builds on. We expect the effect to be dominated by SFT data quantity (dose) more than rank or target modules, and to be config-sensitive: too little SFT gives no advantage, too much may overfit and erode retention.

## Pipelines

P0: Base → GRPO	No SFT. GRPO directly on the base/instruct model.	Can RL alone solve GSM8K? (TinyLoRA says yes, with enough parameters.)

P1: SFT → Stop	SFT only, no GRPO.	What does SFT learn on its own, and what does it cost in retention?

P2: SFT → GRPO	Standard two-stage pipeline.	The net effect of chaining them.

P3: SFT → GRPO (matched params)	Same total trainable parameters as P0, split across stages.	Does the parameter budget matter, or just the order of operations?

## Setup

- Model: Qwen2.5-1.5B-Instruct, LoRA attn r16 alpha 32, seed 42
- Data: SFT dose {500, 5000, 50000}, GRPO on 1500 prompts, capped steps
- Eval: GSM8K exact + format (target), MMLU/IFEval/BBH subsampled (retention)

## Eval

Target benchmarks:
- GSM8K (test accuracy, exact match)
- GSM8K format correctness (does it produce the required reasoning structure?)

Retention benchmarks:
- MMLU: general knowledge and reasoning. The Scalpel vs. Hammer paper shows SFT degrades this more than GRPO.
- IFEval: instruction following. SFT often improves this; GRPO may not.
- BBH: broad reasoning. PEFT-Arena uses this as a “general ability” retention probe.
