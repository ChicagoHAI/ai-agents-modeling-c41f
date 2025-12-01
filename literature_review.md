# Literature Review

## Research Area Overview
Opponent modeling and Theory-of-Mind (ToM) behaviors in multi-agent LLM systems, with a focus on social deduction (Werewolf/Mafia) where agents must infer hidden roles, reason about others’ beliefs, and communicate strategically. Recent work explores LLM-driven agents with rule scaffolding, strategy selection, and emergent ToM signals; classic pre-LLM agents provide non-neural baselines.

## Key Papers

### An Implementation of Werewolf Agent That does not Truly Trust LLMs
- **Authors**: Takehiro Sato, Shintaro Ozaki, Daisaku Yokoyama  
- **Year**: 2024 (arXiv:2409.01575)  
- **Source**: arXiv preprint  
- **Key Contribution**: Hybrid rule + LLM agent that gatekeeps model outputs with templates based on conversation analysis to curb hallucinations and inconsistent persona.  
- **Methodology**: Use LLM to parse dialogue history, then select between rule-based templates or LLM generation conditioned on detected context; includes refusal/end-of-conversation logic.  
- **Datasets Used**: Internal Werewolf simulations (not public); qualitative comparisons.  
- **Results**: Reported improved human-likeness and logical consistency versus pure LLM prompting; qualitative evaluation.  
- **Code Available**: Yes (link in paper; not mirrored here).  
- **Relevance**: Demonstrates safety/control scaffold for social deduction agents and implicit opponent modeling via dialogue analysis.

### Strategy Adaptation in Large Language Model Werewolf Agents
- **Authors**: Fuya Nakamori, Yin Jou Huang, Fei Cheng  
- **Year**: 2025 (arXiv:2507.12732)  
- **Source**: arXiv preprint  
- **Key Contribution**: Explicit strategy-switching controller driven by attitudes toward players and role estimation to adapt LLM behaviors mid-game.  
- **Methodology**: Track conversational sentiment/attitudes; estimate other roles; select among predefined strategies (e.g., aggressive bluff, defensive reveal) and feed strategy-specific prompts to LLM.  
- **Datasets Used**: AIWolf-style simulated games.  
- **Results**: Higher win rate and role inference accuracy than fixed-strategy or implicit prompting baselines; faster convergence on shared beliefs reported.  
- **Code Available**: Noted in paper (not mirrored here).  
- **Relevance**: Directly targets opponent modeling and belief tracking in Werewolf; provides concrete metrics (win rate, inference accuracy).

### A Novel Weighted Ensemble Learning Based Agent for the Werewolf Game
- **Authors**: Mohiuddeen Khan, Claus Aranha  
- **Year**: 2022 (arXiv:2205.09813)  
- **Source**: arXiv preprint  
- **Key Contribution**: Ensemble of pre-existing AIWolf strategies weighted by learned estimates of how other agents perceive the player.  
- **Methodology**: Aggregate outputs of multiple classic agents; weight selection via machine learning to align with opponents’ likely beliefs; no LLMs.  
- **Datasets Used**: AIWolf competition logs.  
- **Results**: Outperforms basic strategies in competitions; improved survival/win rates.  
- **Code Available**: Mentioned but not linked.  
- **Relevance**: Strong non-LLM baseline for opponent modeling; provides dataset context (AIWolf logs).

### Evaluating Large Language Models in Theory of Mind Tasks
- **Authors**: Michal Kosinski  
- **Year**: 2023 (arXiv:2302.02083)  
- **Source**: arXiv preprint  
- **Key Contribution**: Benchmarking ToM capabilities of 11 LLMs via 640 false-belief tasks.  
- **Methodology**: Textual ToM prompts; evaluate accuracy on identifying beliefs of protagonists; statistical comparison across models and versions.  
- **Datasets Used**: Custom ToM prompt set (provided).  
- **Results**: Larger models (e.g., GPT-4) near human-level on many tasks; smaller models weaker.  
- **Code Available**: Prompts included.  
- **Relevance**: Supplies evaluation patterns and metrics for ToM reasoning; adaptable to opponent-model inference questions in games.

### CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society
- **Authors**: Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem  
- **Year**: 2023 (arXiv:2303.17760)  
- **Source**: arXiv preprint  
- **Key Contribution**: Multi-agent role-play framework where paired LLM agents collaborate under role/system prompts, showing emergent coordination/persona modeling.  
- **Methodology**: Role-conditioned conversations with task goals; structured system prompts to reduce drift; measure task success and dialogue quality.  
- **Datasets Used**: Generated dialogues; task-specific prompts.  
- **Results**: Improved task completion vs. single-agent prompting; diverse persona behaviors observed.  
- **Code Available**: Yes (GitHub).  
- **Relevance**: Provides scaffolding for multi-agent orchestration and persona prompts transferable to social deduction experiments.

## Common Methodologies
- **Prompt scaffolding + rules**: Hard constraints or template fallbacks to control LLM outputs in high-stakes turns (2409.01575, 2507.12732).  
- **Strategy/role controllers**: Explicit policy to pick prompts/strategies based on inferred beliefs or attitudes (2507.12732, 2205.09813).  
- **Ensemble / hybrid agents**: Combining multiple agents or modules (rule + LLM or multiple classic bots) to hedge behavior (2205.09813, 2409.01575).  
- **Multi-agent role-play**: Structured roles and interaction patterns to stabilize dialogues (2303.17760).

## Standard Baselines
- Rule-based AIWolf sample agents (seer/villager/werewolf heuristics).  
- Fixed-prompt single-LLM agents without strategy switching.  
- Classic ensemble ML agents (2205.09813).  
- Random or majority-vote vote-selection heuristics in AIWolf logs.

## Evaluation Metrics
- Win rate / survival rate per role (common in AIWolf papers).  
- Hidden role inference accuracy and precision/recall on belief tracking.  
- Dialogue quality/consistency (human-likeness ratings, contradiction counts).  
- Speed of belief convergence (turns to majority consensus) and vote accuracy.  
- ToM task accuracy (false-belief benchmarks like 2302.02083).

## Datasets in the Literature
- **AIWolf competition logs**: Used in 2205.09813; 10k+ games available (downloaded here).  
- **Custom simulated games**: 2409.01575 and 2507.12732 run internal simulations.  
- **ToM prompt sets**: 2302.02083 provides 640 textual tasks.  
- **Generated role-play dialogues**: 2303.17760 produces synthetic multi-agent conversations.

## Gaps and Opportunities
- Limited open evaluations of LLM agents on large real-world social deduction logs.  
- Few quantitative analyses of belief embeddings/trajectory clustering for opponent models.  
- Safety/consistency controls often heuristic; opportunity for probabilistic belief models.  
- Sparse ablations on communication vs. internal reasoning in LLM agents.

## Recommendations for Our Experiment
- **Recommended datasets**: AIWolf 2019 5-player logs (for supervised role/belief inference); Kosinski ToM prompts for auxiliary evaluation; optionally generate structured multi-agent dialogues via CAMEL-style prompts for augmentation.  
- **Recommended baselines**: Rule-based AIWolf sample agents; fixed-prompt single-LLM agent; ensemble ML agent from 2205.09813.  
- **Recommended metrics**: Win rate by role, role inference accuracy/precision, vote alignment with ground truth, contradiction counts in dialogue, convergence speed of group beliefs, and ToM prompt accuracy.  
- **Methodological considerations**: Add strategy-switch controller (per 2507.12732) and LLM-output gatekeeper (per 2409.01575); log internal belief states for embedding analysis; test both dialogue-only and belief-explicit agents to contrast emergent vs. explicit opponent models.
