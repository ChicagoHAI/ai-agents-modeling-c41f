# REPORT: Opponent Modeling in Multi-Agent LLM Werewolf

## 1. Executive Summary
- **Question**: Do LLM agents that maintain explicit opponent models outperform dialogue-only agents at identifying the Werewolf in AIWolf logs?
- **Finding**: On 8 sampled AIWolf 5-player games, the dialogue-only agent hit rate was 0.50 vs 0.38 for the heuristic belief-conditioned agent; no evidence of improvement and trend favored dialogue-only.
- **Implication**: Simple belief priors derived from surface-level accusations can hurt performance; richer belief modeling or stronger base models are needed.

## 2. Goal
- Test whether explicit opponent-model conditioning improves Werewolf identification compared with dialogue-only prompting.
- Importance: Opponent modeling is central to social reasoning; verifying gains clarifies whether explicit belief scaffolds are worthwhile.
- Expected impact: Guidance for agent design in social deduction settings; preliminary baseline for future ToM-style analyses.

## 3. Data Construction
### Dataset Description
- **Source**: AIWolf 2019 5-player competition logs (`datasets/aiwolf_logs/2019final-log05.tar.gz`).
- **Size**: 10k gzipped game logs (6.5 MB); sampled 8 games for this experiment.
- **Format**: CSV-like lines per event with day, type, agent IDs, and utterances.
- **Biases/limits**: Only 5-player setup; agent names are synthetic; language terse and templated.

### Example Samples
```
0,status,3,WEREWOLF,ALIVE,takeda
1,talk,5,1,1,DIVINED Agent[04] WEREWOLF
1,talk,7,1,2,Over
1,talk,10,2,1,VOTE Agent[04]
```

### Data Quality
- Missing values: None observed in parsed fields for sampled logs.
- Outliers: Templated utterances dominate; limited lexical diversity.
- Class distribution: Fixed roles (1 Werewolf, 1 Seer, 1 Possessed, 2 Villagers per game).
- Validation: Parsed roles and talk lines from first 140 rows per game; ensured non-empty roles/talks.

### Preprocessing Steps
1. Streamed gz files directly from tar without full extraction to bound I/O.
2. Parsed `status` lines for role/name maps; parsed `talk` lines into (day, speaker, content).
3. Truncated to first 25 talk turns for prompting to keep context manageable.
4. Built belief heuristics from talk content (vote/divination cues) to produce per-agent p(wolf).

### Train/Val/Test Splits
- No model training; evaluation-only. Sampled 8 distinct games as test set.

## 4. Experiment Description
### Methodology
#### High-Level Approach
Compare two prompting conditions on identical game contexts: (a) dialogue-only LLM inference; (b) LLM conditioned on heuristic belief priors derived from accusations/divinations.

#### Why This Method?
- Mirrors literature on belief-conditioned controllers while keeping setup lightweight.
- Uses real game data to avoid synthetic bias.
- Keeps deterministic decoding (no sampling) for clean paired comparison.

#### Tools and Libraries
- Python 3.12.2, `transformers==4.57.3`, `torch==2.9.1` (CPU), `openai==2.8.1` (unused, no key), `pandas==2.3.3`, `numpy==2.3.5`.

#### Algorithms/Models
- **Dialogue-only**: Qwen2.5-0.5B-Instruct prompted with roles, roster, and early dialogue; asked to output most likely Werewolf.
- **Belief-conditioned**: Same prompt plus heuristic belief table (p_wolf) computed from votes/divinations; instruction to use as prior.
- Heuristic belief update: +0.3 per vote accusation, +0.6 per divination as WEREWOLF, -0.2 per divination as HUMAN; normalized to probabilities.

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| max_games | 8 | Budget-driven (cost/time) |
| talk_turns | 25 | Keep context short |
| temperature | 0.0 | Determinism for comparison |
| model | Qwen2.5-0.5B-Instruct | Small local fallback (no API keys) |

#### Analysis Pipeline
1. Load sample games → build prompts for both conditions.
2. Run paired generations (deterministic) → parse predicted Agent ID.
3. Compute hit if predicted ID matches ground-truth Werewolf.
4. Aggregate accuracy per condition; run McNemar test for paired differences.
5. Plot bar chart of accuracies (`results/plots/accuracy_bar.png`).

### Experimental Protocol
- Runs: 1 per condition per game (deterministic decoding).
- Seeds: Python/NumPy random seeds set to 42.
- Hardware: CPU (no GPU used).
- Execution time: ~2 minutes end-to-end after model download.

### Evaluation Metrics
- **Hit rate (accuracy)**: Whether predicted Agent ID matches true Werewolf (per game). Appropriate for binary success on identification.
- **Paired comparison**: McNemar test for paired correctness across conditions.

### Raw Results
#### Table
| Condition | n games | Accuracy |
|-----------|---------|----------|
| Dialogue-only | 8 | 0.50 |
| Belief-conditioned | 8 | 0.38 |

#### Statistical Test
- McNemar (continuity-corrected) on paired outcomes: n01=0, n10=1, χ²=0.0, p≈1.0 (no significant difference; trend negative for belief prior).

#### Visualizations
- Accuracy bar plot: `results/plots/accuracy_bar.png` (dialogue-only higher).

#### Output Locations
- Metrics: `results/metrics.json`
- Raw generations: `results/llm_outputs.json`

## 5. Result Analysis
### Key Findings
1. Dialogue-only surpassed belief-conditioned (0.50 vs 0.38 hit rate) on 8 games; belief priors did not help.
2. Belief-conditioned never succeeded where dialogue-only failed (n01=0), suggesting heuristic priors skewed decisions rather than correcting them.
3. Errors often traced to over-weighting early mass accusations; small model over-trusted provided belief table instead of dialog nuance.

### Hypothesis Testing Results
- H1/H2 (explicit opponent modeling improves accuracy/convergence): Not supported; belief conditioning reduced accuracy on this small sample (p≈1.0, negligible effect).
- H3 (emergent belief structure) not directly tested; needs embedding analysis (future work).
- H4 (ToM probes) not run due to time/model constraints.

### Comparison to Baselines
- Dialogue-only serves as minimal baseline and outperformed heuristic belief prior.
- No rule-based baseline yet; planned as future extension.

### Error Analysis
- Misclassifications clustered where multiple players accused the same target early; heuristic boosted that target regardless of contradictory evidence.
- Model occasionally misread roles (e.g., interpreting multiple “COMINGOUT SEER” as wolf tells) showing limited understanding of AIWolf conventions.

### Limitations
- Small sample (8 games) and lightweight local model (0.5B) due to unavailable API keys; results are indicative only.
- Heuristic belief modeling simplistic; no temporal decay or role-aware likelihoods.
- Only day-1 style dialogue considered; no full-game rollouts or win-rate simulations.
- No variance estimates across seeds because decoding deterministic.

## 6. Conclusions
- Explicit belief priors, when naïvely derived from surface-level accusations, did not improve Werewolf identification; they slightly harmed accuracy against dialogue-only prompting.
- More structured opponent models (e.g., Bayesian updates over roles, strategy-switch controllers) or stronger LLMs are likely necessary to see benefits.

### Implications
- Opponent modeling must capture conversational reliability, not just accusation counts; priors need calibration.
- Dialogue-only baselines remain competitive when models are small or priors are crude.

### Confidence in Findings
- Low-to-moderate given small sample and small model; directionality (no gain from naive beliefs) is still informative for scaffold design.

## 7. Next Steps
1. Replace heuristic beliefs with probabilistic updates conditioned on role/action likelihoods; re-evaluate with more games.
2. Run with stronger API models (GPT-4.1 / Claude) for better language understanding and ToM probes.
3. Extend evaluation to vote prediction and belief convergence metrics; include rule-based AIWolf agent as non-LLM baseline.
4. Conduct embedding-based clustering of internal belief summaries to test H3 explicitly.

## References
- Sato et al. 2024; Nakamori et al. 2025; Khan & Aranha 2022; Kosinski 2023; Li et al. 2023 (see `papers/`).
