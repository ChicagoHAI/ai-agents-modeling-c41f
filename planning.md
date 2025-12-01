# Research Plan: Opponent Modeling in Multi-Agent LLM Werewolf

## Research Question
Do LLM-based agents in social deduction (Werewolf/AIWolf) develop and exploit mental models of other agents, and do explicit opponent-modeling mechanisms yield measurable gains (win rate, role inference accuracy, belief convergence) over dialogue-only agents?

## Background and Motivation
Social deduction games demand belief tracking, deception, and adaptation—ideal to probe Theory-of-Mind-like behaviors in LLM agents. Prior work (strategy controllers, gating, ensembles) suggests structured belief tracking improves robustness, yet open-source evaluations on AIWolf logs with modern LLMs remain scarce. This project aims to produce reproducible, quantitative evidence using available AIWolf 5-player logs and LLM-based agents.

## Hypothesis Decomposition
- H1: Explicit opponent-modeling agents achieve higher win rates than dialogue-only agents given identical models and sampling settings.
- H2: Explicit belief tracking yields higher hidden-role inference accuracy and earlier belief convergence (fewer turns to consensus) than dialogue-only agents.
- H3: Dialogue-only agents still exhibit emergent belief structure; embeddings of their internal summaries cluster by opponent identity/role more than chance.
- H4 (aux): Agents with explicit models perform better on textual ToM probes than dialogue-only agents after in-game conditioning.

## Proposed Methodology

### Approach
Use AIWolf 2019 5-player logs to create evaluation scenarios and supervised signals for role inference. Implement two LLM agents: (a) Dialogue-only (history-conditioned prompts, no explicit beliefs), (b) Opponent-modeling (maintains per-agent belief vectors over roles, updates via parsed talk/actions, conditions generation on beliefs). Compare via simulated mini-games driven by log-based prompts and via static log reconstruction tasks (predict next vote/role). Complement with ToM prompt set for auxiliary evaluation. Use GPT-4.1 (or available OpenRouter GPT-4.1-equivalent) for generation; use deterministic temperature for fairness.

### Experimental Steps
1. **Data prep**: Sample ~50–100 games from AIWolf logs; parse talk/vote events; extract role labels and turn structure. Split into train/val/test (70/15/15) for supervised probes.
2. **Baseline agent (Dialogue-only)**: Prompt uses full history; generates next action/vote. No explicit belief state; memory is raw dialogue.
3. **Opponent-modeling agent**: Maintain per-agent role belief distribution initialized uniform; update via simple Bayesian-like heuristic from events; pass belief summary into prompt; also request model to state current belief before action (logged for analysis).
4. **Static evaluation**: Given partial game history, predict hidden roles (classification) and next vote target; compute accuracy/F1.
5. **Simulation evaluation**: Run few-turn rollouts using log segments as seeds; have agents propose votes; score alignment with ground-truth majority vote and survival odds proxy.
6. **ToM auxiliary**: Run Kosinski-style false-belief subset (e.g., 50 items) with/without belief-conditioning to see any shift.
7. **Embedding analysis**: Embed agent belief summaries (per player per turn) using model embeddings; cluster by true role; measure silhouette/ARI against random baseline.
8. **Statistics**: Paired comparison of agents on shared test splits; bootstrap CIs for accuracy and convergence turns; McNemar for paired classification; paired t-test or Wilcoxon on convergence steps.

### Baselines
- Rule-based heuristics from AIWolf sample agents (vote majority heuristic) for static prediction.
- Dialogue-only LLM agent (our baseline).
- Optional: majority-class predictor for role inference.

### Evaluation Metrics
- Win-rate proxy: vote alignment with ground-truth lynch outcome (%), survival probability proxy (werewolf avoided, seer survives past day 2).
- Role inference: accuracy, macro-F1 across roles (WEREWOLF, SEER, VILLAGER, POSSESSED).
- Belief convergence: turns until belief entropy < threshold or majority >0.6; compare means.
- ToM probe accuracy on subset.
- Embedding clustering: silhouette score vs random baseline.

### Statistical Analysis Plan
- Use paired McNemar for classification differences (role inference).
- Bootstrap 1,000 resamples for CIs on metrics (win proxy, belief convergence steps, ToM accuracy).
- Wilcoxon signed-rank for convergence steps if non-normal; otherwise paired t-test after Shapiro test.
- Report effect sizes (Cohen's d or Cliff's delta) where applicable; significance at α=0.05.

## Expected Outcomes
- Support H1/H2 if opponent-modeling agent shows statistically significant gains on win proxy and role inference with faster convergence.
- H3 supported if embedding clustering > random baseline with positive silhouette.
- H4 supported if belief-conditioning improves ToM probe accuracy.

## Timeline and Milestones
- Data prep & EDA: 45 min
- Implement agents/prompts & parsing: 60 min
- Run evaluations (static + limited rollouts + ToM subset): 60–90 min
- Analysis & plots: 45 min
- Reporting (REPORT.md, README.md): 45 min

## Potential Challenges
- Parsing AIWolf logs robustly; mitigate with small sample and schema checks.
- API cost/latency: keep sample small (<=100 games, short turns) and cache responses.
- Prompt sensitivity: may need quick prompt iteration; log prompts/temps.
- Simulation realism: limited by non-interactive logs; use rollouts seeded from logs to bound behavior.

## Success Criteria
- Reproducible pipeline with documented prompts and seeds.
- At least one statistically compared metric (role inference or belief convergence) between agents with CIs.
- REPORT.md containing actual model outputs/metrics and limitations.
