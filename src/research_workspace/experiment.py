import gzip
import json
import os
import random
import re
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class TalkTurn:
    day: int
    speaker: int
    content: str


@dataclass
class GameSample:
    game_id: str
    roles: Dict[int, str]
    names: Dict[int, str]
    talks: List[TalkTurn]


def load_games(tar_path: str, max_games: int = 10, max_lines: int = 120) -> List[GameSample]:
    """Stream a limited number of games from the AIWolf tarball."""
    samples: List[GameSample] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".log.gz"):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            raw = fileobj.read()
            sample = parse_log(raw, member.name, max_lines=max_lines)
            if sample:
                samples.append(sample)
            if len(samples) >= max_games:
                break
    return samples


def parse_log(raw_bytes: bytes, game_id: str, max_lines: int = 120) -> Optional[GameSample]:
    """Parse a single log (gzipped) into structured roles and talk turns."""
    text = gzip.decompress(raw_bytes).decode("utf-8", errors="ignore").splitlines()
    roles: Dict[int, str] = {}
    names: Dict[int, str] = {}
    talks: List[TalkTurn] = []
    for idx, line in enumerate(text):
        if idx >= max_lines:
            break
        parts = line.split(",", maxsplit=5)
        if len(parts) < 3:
            continue
        day_raw, event_type = parts[0], parts[1]
        if not day_raw.isdigit():
            continue
        day = int(day_raw)
        if event_type == "status" and len(parts) >= 6:
            agent_id = int(parts[2])
            role = parts[3]
            name = parts[5]
            roles[agent_id] = role
            names[agent_id] = name
        elif event_type == "talk" and len(parts) >= 6:
            speaker = int(parts[4])
            content = parts[5].strip()
            talks.append(TalkTurn(day=day, speaker=speaker, content=content))
    if not roles or not talks:
        return None
    return GameSample(game_id=game_id, roles=roles, names=names, talks=talks)


def suspicion_scores(game: GameSample) -> Dict[int, float]:
    """Heuristic belief updater: count accusations and hostile divinations."""
    suspicion = {agent: 0.0 for agent in game.roles.keys()}
    vote_pattern = re.compile(r"VOTE Agent\[(\d+)]")
    divine_wolf_pattern = re.compile(r"DIVINED Agent\[(\d+)] WEREWOLF")
    divine_human_pattern = re.compile(r"DIVINED Agent\[(\d+)] HUMAN")
    for talk in game.talks:
        for target in vote_pattern.findall(talk.content):
            suspicion[int(target)] += 0.3
        for target in divine_wolf_pattern.findall(talk.content):
            suspicion[int(target)] += 0.6
        for target in divine_human_pattern.findall(talk.content):
            suspicion[int(target)] -= 0.2
    # Normalize to probabilities (softmax-style, keep positive masses).
    vals = np.array(list(suspicion.values()), dtype=float)
    vals = vals - vals.min() + 1e-3  # shift to positive
    probs = vals / vals.sum()
    return {agent: float(prob) for agent, prob in zip(suspicion.keys(), probs)}


def build_context(game: GameSample, max_turns: int = 25) -> str:
    """Format dialogue history for prompting."""
    lines = []
    for turn in game.talks[:max_turns]:
        name = game.names.get(turn.speaker, f"Agent[{turn.speaker:02d}]")
        lines.append(f"Day {turn.day} | Agent[{turn.speaker:02d}] ({name}): {turn.content}")
    return "\n".join(lines)


def build_prompt(game: GameSample, condition: str, beliefs: Optional[Dict[int, float]] = None) -> List[Dict[str, str]]:
    """Construct chat messages for the specified condition."""
    role_intro = "Roles: 1 WEREWOLF, 1 SEER, 1 POSSESSED, 2 VILLAGER. Player IDs are Agent[01]..Agent[05]."
    roster = ", ".join([f"Agent[{aid:02d}]={name}" for aid, name in sorted(game.names.items())])
    history = build_context(game)
    system = (
        "You are an analyst for AIWolf (Werewolf). Infer which agent is most likely the WEREWOLF from early dialogue. "
        "Respond with the single Agent ID (e.g., Agent[03]) and a concise rationale."
    )
    user_parts = [role_intro, f"Roster: {roster}", "Dialogue (chronological):", history]
    if condition == "beliefs" and beliefs:
        belief_str = "; ".join([f"Agent[{aid:02d}]: p_wolf={prob:.2f}" for aid, prob in sorted(beliefs.items())])
        user_parts.append(f"Belief model (heuristic from votes/divinations): {belief_str}")
        user_parts.append("Use the belief scores as prior but override if dialogue strongly contradicts them.")
    user_parts.append("Which agent is most likely the WEREWOLF? Reply with Agent[NN] then one-sentence justification.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


class ModelCaller:
    """Unified interface for OpenAI API or local HF model."""

    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openai_key:
            self.mode = "openai"
            self.model = "gpt-4.1"
            self.client = OpenAI(api_key=openai_key)
        elif openrouter_key:
            self.mode = "openai"
            self.model = "gpt-4.1"
            self.client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        else:
            self.mode = "local"
            self.model = "Qwen/Qwen2.5-0.5B-Instruct"
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.model,
                device_map="cpu",
                torch_dtype="auto",
            )

    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        if self.mode == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=120,
            )
            return resp.choices[0].message.content.strip()
        # Local generation: flatten chat messages into a single prompt.
        prompt_lines = []
        for msg in messages:
            role = msg["role"].upper()
            prompt_lines.append(f"{role}: {msg['content']}")
        prompt_lines.append("ASSISTANT:")
        prompt = "\n".join(prompt_lines)
        outputs = self.generator(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"][len(prompt) :].strip()


def extract_prediction(text: str) -> Optional[int]:
    """Parse Agent ID from model response."""
    match = re.search(r"Agent\[(\d{2})]", text)
    if match:
        return int(match.group(1))
    return None


def evaluate_conditions(samples: List[GameSample], model_caller: ModelCaller):
    """Run both conditions and compute accuracy."""
    results = []
    for game in tqdm(samples, desc="Running LLM conditions"):
        wolf_ids = {aid for aid, role in game.roles.items() if role == "WEREWOLF"}
        beliefs = suspicion_scores(game)
        contexts = {
            "dialogue_only": build_prompt(game, "dialogue"),
            "beliefs": build_prompt(game, "beliefs", beliefs=beliefs),
        }
        for condition, msgs in contexts.items():
            try:
                output = model_caller(msgs)
            except Exception as exc:  # pylint: disable=broad-except
                results.append(
                    {
                        "game_id": game.game_id,
                        "condition": condition,
                        "error": str(exc),
                    }
                )
                continue
            pred = extract_prediction(output)
            hit = bool(pred in wolf_ids) if pred is not None else False
            results.append(
                {
                    "game_id": game.game_id,
                    "condition": condition,
                    "prediction": pred,
                    "wolf_ids": sorted(list(wolf_ids)),
                    "hit": hit,
                    "response": output,
                }
            )
    return results


def aggregate_metrics(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate accuracy per condition."""
    metrics: Dict[str, Dict[str, float]] = {}
    by_condition: Dict[str, List[bool]] = {}
    for row in rows:
        if "hit" not in row:
            continue
        by_condition.setdefault(row["condition"], []).append(bool(row["hit"]))
    for condition, hits in by_condition.items():
        acc = float(np.mean(hits)) if hits else 0.0
        metrics[condition] = {"n": len(hits), "accuracy": acc}
    return metrics


def main():
    tar_path = "datasets/aiwolf_logs/2019final-log05.tar.gz"
    samples = load_games(tar_path, max_games=8, max_lines=140)
    model_caller = ModelCaller()
    results = evaluate_conditions(samples, model_caller)
    metrics = aggregate_metrics(results)

    os.makedirs("results", exist_ok=True)
    with open("results/llm_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print concise summary for quick inspection.
    print("Model:", model_caller.model, "| mode:", model_caller.mode)
    for condition, vals in metrics.items():
        print(f"{condition}: n={vals['n']} accuracy={vals['accuracy']:.2f}")


if __name__ == "__main__":
    main()
