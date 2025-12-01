# Resources Catalog

## Summary
Catalog of papers, datasets, and code gathered for studying opponent modeling/Theory-of-Mind in multi-agent LLMs playing social deduction games.

## Papers
Total papers downloaded: 5

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| An Implementation of Werewolf Agent That does not Truly Trust LLMs | Sato, Ozaki, Yokoyama | 2024 | papers/2409.01575_llm_werewolf_trust.pdf | Rule-based gatekeeper for LLM outputs in Werewolf agents |
| Strategy Adaptation in Large Language Model Werewolf Agents | Nakamori, Huang, Cheng | 2025 | papers/2507.12732_strategy_adaptation_llm_werewolf.pdf | Strategy switching based on inferred roles/attitudes |
| A Novel Weighted Ensemble Learning Based Agent for the Werewolf Game | Khan, Aranha | 2022 | papers/2205.09813_weighted_werewolf_agent.pdf | Ensemble baseline estimating others’ beliefs |
| Evaluating Large Language Models in Theory of Mind Tasks | Kosinski | 2023 | papers/2302.02083_theory_of_mind_llms.pdf | 640-task ToM benchmark for LLMs |
| CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society | Li et al. | 2023 | papers/2303.17760_camel.pdf | Multi-agent role-play framework showing emergent coordination |

See papers/README.md for details.

## Datasets
Total datasets downloaded: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| AIWolf 2019 5-Player Game Logs | http://aiwolf.org/archive/2019final-log05.tar.gz | 6.5 MB (10k gzipped logs) | Social deduction dialogue/action | datasets/aiwolf_logs/2019final-log05.tar.gz | Sample snippet in `samples/000_head.txt`; decompress per README |

See datasets/README.md for download/inspection instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| AIWolfPy | https://github.com/aiwolf/AIWolfPy | Official Python AIWolf client & sample agents | code/aiwolfpy | Useful for sim/playback and baseline agents |
| aiwolf-nlp-agent | https://github.com/aiwolfdial/aiwolf-nlp-agent | NLP-focused AIWolf agent pipeline | code/aiwolf-nlp-agent | Potential LLM integration scaffold |

See code/README.md for more context.

## Resource Gathering Notes

**Search Strategy**: Queried arXiv for Werewolf/LLM/ToM keywords; pulled recent social deduction LLM papers; located AIWolf competition logs via official site; identified AIWolf codebases via GitHub search.  
**Selection Criteria**: Prioritized direct social deduction agents, ToM evaluations, multi-agent communication scaffolds, and resources with accessible code/data.  
**Challenges Encountered**: arXiv keyword search noisy; relied on targeted IDs for Werewolf/ToM papers. AIWolf log links are large; selected the smaller 5-player archive for practicality.  
**Gaps and Workarounds**: Few open-source LLM Werewolf agents with code; leveraged generic AIWolf NLP agent as scaffold and classic baseline papers for comparison.

## Recommendations for Experiment Design
1. **Primary dataset(s)**: AIWolf 2019 5-player logs for supervised belief/role inference; augment with ToM prompt tasks for auxiliary evaluation.
2. **Baseline methods**: Rule-based AIWolf sample agents and ensemble baseline (2205.09813); fixed-prompt single-LLM agent; CAMEL-style role-play without explicit opponent model.
3. **Evaluation metrics**: Win/survival rate by role, role inference accuracy/precision, vote alignment, contradiction counts, belief convergence speed, ToM task accuracy.
4. **Code to adapt/reuse**: AIWolfPy for environment/playback; aiwolf-nlp-agent for language processing pipeline; implement strategy-switch and gating modules inspired by 2507.12732 and 2409.01575.
