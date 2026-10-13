"""Historical synthesis prompts and deterministic dataset merge only.

The unfinished Claude generation path is disabled. AtlasLex uses Gemini only;
this utility makes no external API requests and requires no Anthropic package.
"""

import argparse
import json
import random
from pathlib import Path

GENERATION_PROMPTS = [
    # Type A: Direct + practical
    """Génère 5 questions-réponses pour l'article {n} du Code de Commerce marocain.
Contexte : {context}
Article {n} : {text}

Règles :
- 2 questions directes sur l'article
- 2 scénarios pratiques d'un commerçant ou justiciable
- 1 question de comparaison ou de synthèse avec d'autres dispositions
- Chaque réponse cite explicitement "Article {n}" et "Loi n° 15-95"
- Réponses complètes (100-300 mots)

JSON format:
[{{"question": "...", "reponse": "...", "type": "direct|scenario|synthese"}}]""",
    # Type B: Edge cases + exceptions
    """Pour l'article {n} du Code de Commerce marocain :
Contexte : {context}
Texte : {text}

Génère 4 questions-réponses axées sur :
- Les conditions d'application et exceptions
- Les délais, sanctions ou effets juridiques mentionnés
- Les cas limites ou ambiguïtés pratiques
- La procédure à suivre pour un praticien

JSON: [{{"question": "...", "reponse": "...", "type": "procedure|sanction|condition|exception"}}]""",
    # Type C: Cross-article synthesis
    """Article {n} (Code de Commerce marocain) :
Contexte : {context}
Texte : {text}
Articles liés : {cross_refs}

Génère 3 questions de synthèse qui nécessitent de combiner plusieurs articles.
Réponses structurées avec sous-titres si nécessaire.

JSON: [{{"question": "...", "reponse": "...", "type": "synthese"}}]""",
]


def split_examples(examples: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Keep every connected set of article references in one partition."""
    parent = {}

    def root(ref):
        parent.setdefault(ref, ref)
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    for example in examples:
        refs = [str(r) for r in example.get("article_refs", [])]
        for ref in refs[1:]:
            parent[root(ref)] = root(refs[0])
    groups = {}
    for index, example in enumerate(examples):
        refs = example.get("article_refs", [])
        key = root(str(refs[0])) if refs else f"unreferenced-{index}"
        groups.setdefault(key, []).append(example)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("At least two independent article groups are required for a split.")
    eval_keys = set(keys[: max(1, round(len(keys) * 0.05))])
    train = [e for key in keys if key not in eval_keys for e in groups[key]]
    evaluation = [e for key in keys if key in eval_keys for e in groups[key]]
    return train, evaluation


def merge_all_datasets(output_dir: Path, seed: int = 42) -> tuple[int, int]:
    examples, seen = [], set()
    for name in ["train_chatml.jsonl", "synthetic_chatml.jsonl"]:
        path = output_dir / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            example = json.loads(line)
            if not isinstance(example, dict) or not isinstance(example.get("messages"), list):
                raise ValueError(f"Invalid example in {name}.")
            key = json.dumps(example, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                examples.append(example)
    train, evaluation = split_examples(examples, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("final_train.jsonl", train), ("final_eval.jsonl", evaluation)]:
        (output_dir / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    return len(train), len(evaluation)


if __name__ == "__main__":
    from atlaslex.config import project_home

    parser = argparse.ArgumentParser(
        description="Merge existing datasets; synthetic API generation is disabled"
    )
    parser.add_argument("--output", type=Path, default=project_home() / "data/processed/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not args.merge_only:
        parser.error(
            "Synthetic API generation is unfinished and disabled. Use --merge-only for existing datasets."
        )
    try:
        train, evaluation = merge_all_datasets(args.output, args.seed)
        print(f"Train: {train}; evaluation: {evaluation}. No training was performed.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
