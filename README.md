# Opponent Modeling in AIWolf (Werewolf) Agents

## Overview
Short study of dialogue-only vs belief-conditioned LLM agents on AIWolf 5-player logs to see if explicit opponent modeling helps identify the Werewolf. Using a small local model (Qwen2.5-0.5B-Instruct), dialogue-only outperformed the heuristic belief prior on an 8-game sample.

## Key Findings
- Dialogue-only hit rate: 0.50 (4/8); belief-conditioned: 0.38 (3/8).
- Belief priors based purely on accusation/divination counts never corrected a dialogue-only miss.
- Heuristic priors likely overweight early mass accusations; richer belief calibration needed.

## Reproduction
1. Create environment (already configured here): `uv venv && source .venv/bin/activate`.
2. Install deps: `uv sync` (uses `pyproject.toml`).
3. Run experiment: `python -m research_workspace.experiment` (downloads Qwen2.5-0.5B if no API keys; uses GPT-4.1 if `OPENAI_API_KEY`/`OPENROUTER_API_KEY` set).
4. Outputs:
   - Metrics: `results/metrics.json`
   - Raw generations: `results/llm_outputs.json`
   - Plot: `results/plots/accuracy_bar.png`

## File Structure
- `planning.md` – research plan
- `src/research_workspace/experiment.py` – data parsing, prompting, evaluation
- `datasets/` – AIWolf logs + README
- `results/` – experiment outputs and plots
- `REPORT.md` – full report with methodology and analysis

See `REPORT.md` for detailed methods, limitations, and next steps.
