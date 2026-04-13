"""
run_experiments.py  —  Ablation Experiment Runner

Runs a full matrix of experiments: models × defense_configs × sequences.
For each combination, ConversationRunner runs the sequence and score_result()
scores the output. Results are saved to a timestamped JSON in results/.

Usage:
  python run_experiments.py
  python run_experiments.py --models groq/llama-3.1-8b-instant --defenses none --limit 3
  python run_experiments.py --models groq/llama-3.1-8b-instant groq/llama-3.3-70b-versatile
"""

import json
import time
import datetime
import logging
import random
import argparse
from pathlib import Path
from collections import defaultdict

from core.config import TARGET_MODELS, DEFENSE_CONFIGS, RANDOM_SEED, RESULTS_DIR, DATA_DIR
from core.runner import ConversationRunner

try:
    from scoring.scorer import score_result
except ImportError:
    def score_result(result):
        result["attack_success_rate"] = None
        result["failure_mode"]        = None
        result["over_refusal"]        = False
        result["detection_latency_turns"] = result.get("defense_triggered_at_turn")
        result["scoring"] = {"note": "Scorer not available — run scoring separately"}
        return result

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

random.seed(RANDOM_SEED)

ATTACK_FILE = DATA_DIR / "attack_sequences" / "sequences.json"
BENIGN_FILE = DATA_DIR / "benign_sequences"  / "benign.json"
OUT_DIR     = Path(RESULTS_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    # JSON files wrap sequences in a dict: {"sequences": [...]} or {"conversations": [...]}
    if isinstance(data, dict):
        return data.get("sequences") or data.get("conversations") or []
    return data


def estimate_runtime_minutes(models, defenses, n_sequences) -> float:
    """Estimates experiment duration accounting for per-turn and per-pair rate-limit delays."""
    total_seconds = 0
    avg_turns = 5
    for mk in models:
        delay     = TARGET_MODELS[mk]["rate_limit_delay"]
        is_gemini = TARGET_MODELS[mk]["provider"] == "gemini"
        for _ in defenses:
            total_seconds += n_sequences * avg_turns * delay
            total_seconds += 60 if is_gemini else 30
    return total_seconds / 60


def run_experiment(
    models:   list[str] | None = None,
    defenses: list[str] | None = None,
    limit:    int | None       = None,
) -> tuple[list, str]:
    """
    Run the full ablation experiment matrix.

    Args:
        models:   list of model keys from TARGET_MODELS. None = all.
        defenses: list of defense keys from DEFENSE_CONFIGS. None = all.
        limit:    max number of sequences per (model, defense) pair.

    Returns:
        (all_results, output_file_path)
    """
    models_to_run   = models   or list(TARGET_MODELS.keys())
    defenses_to_run = defenses or list(DEFENSE_CONFIGS.keys())

    attack_seqs = load_json(ATTACK_FILE)
    benign_seqs = load_json(BENIGN_FILE)

    # Tag benign sequences so scorer treats them as over-refusal checks
    for s in benign_seqs:
        s["attack_pattern"] = "benign"
        s["harm_category"]  = "benign"

    all_sequences = attack_seqs + benign_seqs
    if limit:
        all_sequences = all_sequences[:limit]

    n_attack = sum(1 for s in all_sequences if s.get("attack_pattern") != "benign")
    n_benign = sum(1 for s in all_sequences if s.get("attack_pattern") == "benign")
    n_total  = len(all_sequences)

    run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    est_mins = estimate_runtime_minutes(models_to_run, defenses_to_run, n_total)

    print("\n" + "="*65)
    print(f"  SafeBench AI — Ablation Experiment {run_id}")
    print("="*65)
    print(f"  Models:    {[TARGET_MODELS[m]['display_name'] for m in models_to_run]}")
    print(f"  Defenses:  {[DEFENSE_CONFIGS[d]['label'] for d in defenses_to_run]}")
    print(f"  Sequences: {n_total} total  ({n_attack} attack, {n_benign} benign)")
    print(f"  API calls: ~{n_total * len(models_to_run) * len(defenses_to_run) * 5}")
    print(f"  Est. time: ~{est_mins:.0f} minutes (rate-limit delays included)")
    print("="*65)

    all_results = []

    for pair_idx, model_key in enumerate(models_to_run):
        for def_idx, defense_key in enumerate(defenses_to_run):

            m_label = TARGET_MODELS[model_key]["display_name"]
            d_label = DEFENSE_CONFIGS[defense_key]["label"]

            print(f"\n{'-'*65}")
            print(f"  Running: [{m_label}] × [{d_label}]")
            print(f"{'-'*65}")

            runner = ConversationRunner(model_key, defense_key)

            for seq_idx, sequence in enumerate(all_sequences):
                seq_id    = sequence.get("id", f"seq_{seq_idx}")
                is_attack = sequence.get("attack_pattern") != "benign"
                tag       = "ATK" if is_attack else "BEN"

                print(f"  [{tag}] {seq_id:<12}", end="", flush=True)

                try:
                    raw_result    = runner.run_sequence(sequence)
                    scored_result = score_result(raw_result)
                    all_results.append(scored_result)

                    asr     = scored_result.get("attack_success_rate")
                    lat     = scored_result.get("detection_latency_turns")
                    refusal = scored_result.get("over_refusal", False)

                    if is_attack:
                        if asr is None:
                            status = "ran (not scored)"
                        elif asr == 1:
                            lat_str = f" (detected turn {lat})" if lat else ""
                            status  = f"BREACHED {lat_str}"
                        else:
                            status = "blocked"
                        print(f"  {status}")
                    else:
                        print(f"  {'over-refused' if refusal else 'OK'}")

                except Exception as e:
                    logger.error(f"Error on {seq_id}: {e}")
                    print(f"  ERROR: {e}")

            is_last_pair = (
                def_idx == len(defenses_to_run) - 1 and
                pair_idx == len(models_to_run) - 1
            )
            if not is_last_pair:
                provider = TARGET_MODELS[model_key]["provider"]
                pause    = 60 if provider == "gemini" else 30
                print(f"\n  [Pausing {pause}s to reset rate-limit window...]")
                time.sleep(pause)

    out_path = OUT_DIR / f"run_{run_id}.json"
    payload  = {
        "run_id":    run_id,
        "timestamp": run_id,
        "seed":      RANDOM_SEED,
        "models":    models_to_run,
        "defenses":  defenses_to_run,
        "n_attack":  n_attack,
        "n_benign":  n_benign,
        "results":   all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Done! Results saved to: {out_path}")
    print(f"{'='*65}")

    _print_summary(all_results)
    return all_results, str(out_path)


def _print_summary(results: list):
    """Prints the ablation table and over-refusal rates to the terminal."""
    attack = [r for r in results if r.get("attack_pattern") != "benign"]
    benign = [r for r in results if r.get("attack_pattern") == "benign"]

    if not any(r.get("attack_success_rate") is not None for r in attack):
        print("\n  (Scoring not available — run scoring separately)")
        return

    print("\n  ABLATION SUMMARY — ASR by model and defense")
    print("  " + "-"*63)
    print(f"  {'Model':<30} {'Defense':<30} {'ASR':>5}")
    print("  " + "-"*63)

    groups: dict = defaultdict(list)
    for r in attack:
        if r.get("attack_success_rate") is not None:
            key = (r.get("model_key",""), r.get("defense_key",""))
            groups[key].append(r["attack_success_rate"])

    defense_order = ["none", "system_prompt", "system_prompt_plus_classifier"]
    for model_key in set(k[0] for k in groups):
        prev_asr = None
        for def_key in defense_order:
            vals = groups.get((model_key, def_key))
            if not vals:
                continue
            asr      = sum(vals) / len(vals)
            reduction = f"  (-{prev_asr - asr:.0%})" if prev_asr is not None else ""
            m_name   = TARGET_MODELS.get(model_key, {}).get("display_name", model_key)
            d_name   = DEFENSE_CONFIGS.get(def_key, {}).get("label", def_key)
            print(f"  {m_name:<30} {d_name:<30} {asr:>5.0%}{reduction}")
            prev_asr = asr
        print()

    print("  OVER-REFUSAL RATE (benign sequences incorrectly refused)")
    print("  " + "-"*63)
    or_groups: dict = defaultdict(list)
    for r in benign:
        key = (r.get("model_key",""), r.get("defense_key",""))
        or_groups[key].append(1 if r.get("over_refusal") else 0)
    for (mk, dk), vals in sorted(or_groups.items()):
        rate   = sum(vals) / len(vals)
        m_name = TARGET_MODELS.get(mk, {}).get("display_name", mk)
        d_name = DEFENSE_CONFIGS.get(dk, {}).get("label", dk)
        print(f"  {m_name:<30} {d_name:<30} {rate:>5.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SafeBench ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --models groq/llama-3.1-8b-instant --defenses none --limit 3
  python run_experiments.py --models groq/llama-3.1-8b-instant
  python run_experiments.py
        """
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(TARGET_MODELS.keys()),
        help="Model keys to test (default: all)"
    )
    parser.add_argument(
        "--defenses",
        nargs="+",
        choices=list(DEFENSE_CONFIGS.keys()),
        help="Defense keys to test (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max sequences per run — useful for quick tests"
    )
    args = parser.parse_args()

    run_experiment(
        models   = args.models,
        defenses = args.defenses,
        limit    = args.limit,
    )
