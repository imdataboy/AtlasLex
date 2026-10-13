"""Optional, rule-based ChatML/Alpaca export from a legal PDF. No model training.

These are source-derived templates, not a measured or human-validated dataset.
"""

import json
import random
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import argparse


SYSTEM_PROMPT = (
    "Tu es un expert juridique spécialisé dans le Code de Commerce marocain "
    "(Loi n° 15-95). Tu réponds avec précision et clarté, en citant toujours "
    "les articles pertinents. Si une question dépasse le périmètre du Code de "
    "Commerce marocain, tu l'indiques clairement. Tes réponses sont en français."
)


@dataclass
class TrainingExample:
    instruction: str
    input_context: str
    output: str
    article_refs: list[str]
    category: str
    source: str


# Question templates

DIRECT_TEMPLATES = [
    "Que dit l'article {n} du Code de Commerce marocain ?",
    "Quel est le contenu de l'article {n} de la loi n° 15-95 ?",
    "Expliquez l'article {n} du Code de Commerce.",
    "Que prévoit l'article {n} du Code de Commerce marocain ?",
    "Quelles sont les dispositions de l'article {n} ?",
]

# Semantic templates teach the model what an article *does*
SEMANTIC_TEMPLATES = [
    "Quelles obligations l'article {n} impose-t-il aux commerçants ?",
    "Dans quelles situations l'article {n} du Code de Commerce s'applique-t-il ?",
    "Quel problème juridique l'article {n} cherche-t-il à résoudre ?",
    "Quel est l'objectif juridique de l'article {n} du Code de Commerce marocain ?",
    "Quelle règle juridique l'article {n} établit-il ?",
]

# Article lookup templates teach "which article covers X"
ARTICLE_LOOKUP_TEMPLATES = [
    "Quel article du Code de Commerce traite de la règle suivante : {snippet} ?",
    "Dans quel article du Code de Commerce trouve-t-on la disposition suivante : {snippet} ?",
    "Quel article du Code de Commerce marocain contient cette règle : {snippet} ?",
]

# RAG-style templates teach answering from injected context
RAG_TEMPLATES = [
    "Selon l'extrait suivant du Code de Commerce, quelle règle est établie ?",
    "En vous basant sur ce texte juridique, quelle obligation est imposée ?",
    "D'après cet article du Code de Commerce, que peut-on conclure ?",
    "Analysez cet extrait du Code de Commerce marocain et expliquez son application.",
]

LIVRE_CONTEXT_TEMPLATES = [
    "Dans le cadre du {livre}, que prévoit l'article {n} ?",
    "Concernant {chapitre}, quel est l'objet de l'article {n} ?",
]

CROSS_REF_TEMPLATES = [
    "Quels articles du Code de Commerce sont liés à l'article {n} et comment s'articulent-ils ?",
    "L'article {n} fait référence aux articles {refs}. Expliquez le lien entre ces dispositions.",
]

# Out-of-scope questions teach the model to refuse gracefully
OUT_OF_SCOPE_QUESTIONS = [
    ("Quelle est la peine pour vol simple selon le Code de Commerce ?", "Code pénal"),
    ("Comment obtenir la nationalité marocaine ?", "droit de la nationalité"),
    ("Quelles sont les règles du divorce au Maroc ?", "Code de la famille (Moudawwana)"),
    ("Quelles sont les conditions pour licencier un employé au Maroc ?", "Code du travail"),
    ("Comment contester un redressement fiscal au Maroc ?", "droit fiscal"),
    ("Quelles sont les règles d'urbanisme pour construire au Maroc ?", "droit de l'urbanisme"),
    ("Comment déposer une plainte pénale contre un associé ?", "Code de procédure pénale"),
    (
        "Quelles sont les règles d'héritage au Maroc ?",
        "droit successoral islamique et Code de la famille",
    ),
    ("Comment immatriculer un véhicule au Maroc ?", "réglementation administrative"),
    (
        "Quelles sont les conditions pour obtenir un visa de travail au Maroc ?",
        "droit de l'immigration et du travail",
    ),
]


# Answer builders───


def make_direct_answer(article: dict) -> str:
    """
    IMPROVED: Three-section format — explanation + quoted text + reference.
    Teaches the model to explain, not just copy-paste the law.
    """
    ctx = article["context_header"]
    art_num = article["article_number"]
    text = article["text"]

    location = f"Situé dans : {ctx}\n\n" if ctx else ""

    answer = (
        f"{location}"
        f"Explication juridique :\n"
        f"L'article {art_num} du Code de Commerce marocain établit les dispositions "
        f"suivantes dans le cadre de {ctx}.\n\n"
        f"Texte de l'article :\n"
        f"« {text} »\n\n"
        f"Référence juridique : Art. {art_num}, Loi n° 15-95 formant Code de Commerce."
    )

    if article["footnotes"]:
        answer += "\n\nNotes :\n" + "\n".join(article["footnotes"])

    return answer


def make_cross_ref_answer(article: dict, all_articles: dict) -> str:
    """Generate an answer explaining cross-referenced articles."""
    art_num = article["article_number"]
    refs = article["cross_refs"][:3]

    answer = f"L'article {art_num} s'articule avec les dispositions suivantes :\n\n"
    answer += f"**Article {art_num}** : {article['text']}\n\n"

    for ref_num in refs:
        ref_art = all_articles.get(ref_num)
        if ref_art:
            answer += (
                f"**Article {ref_num}** (cité dans l'article {art_num}) : {ref_art['text']}\n\n"
            )

    answer += f"\n[Références : Art. {art_num}"
    if refs:
        answer += ", Art. " + ", Art. ".join(str(r) for r in refs)
    answer += ", Loi n° 15-95]"
    return answer


def make_lookup_answer(article: dict) -> str:
    """Answer for 'which article covers this?' questions."""
    art_num = article["article_number"]
    ctx = article["context_header"]
    return (
        f"Cette disposition se trouve dans l'article {art_num} "
        f"du Code de Commerce marocain (Loi n° 15-95).\n\n"
        f"Contexte : {ctx}\n\n"
        f"[Référence : Art. {art_num}, Loi n° 15-95]"
    )


def make_rag_answer(article: dict) -> str:
    """Answer for RAG-style context questions."""
    art_num = article["article_number"]
    ctx = article["context_header"]
    text = article["text"]

    return (
        f"D'après l'article {art_num} du Code de Commerce marocain ({ctx}), "
        f"la règle établie est la suivante :\n\n"
        f"« {text} »\n\n"
        f"Cette disposition s'applique dans le cadre du droit commercial marocain "
        f"et doit être lue avec les éventuelles notes de modification ou d’abrogation.\n\n"
        f"[Référence : Art. {art_num}, Loi n° 15-95]"
    )


# Scenario templates

SCENARIO_TEMPLATES = {
    "LIVRE PREMIER": [
        {
            "q": "Un ressortissant étranger souhaite exercer une activité commerciale au Maroc. Quelles sont les conditions légales à respecter selon le Code de Commerce ?",
            "hint_articles": [],
        },
        {
            "q": "Un commerçant ne s'est pas inscrit au registre du commerce depuis l'ouverture de son activité. Quelles sont les conséquences juridiques prévues par le Code de Commerce ?",
            "hint_articles": [],
        },
        {
            "q": "Quelles obligations comptables le Code de Commerce impose-t-il aux commerçants ?",
            "hint_articles": [],
        },
    ],
    "LIVRE II": [
        {
            "q": "Un commerçant souhaite vendre son fonds de commerce. Quelles sont les formalités légales obligatoires selon le Code de Commerce marocain ?",
            "hint_articles": [],
        },
        {
            "q": "Quelle est la différence entre la vente du fonds de commerce et son nantissement ? Quelles dispositions du Code de Commerce s'appliquent ?",
            "hint_articles": [],
        },
        {
            "q": "Un propriétaire souhaite donner son fonds de commerce en gérance libre. Quelles conditions et obligations prévoit le Code de Commerce ?",
            "hint_articles": [],
        },
    ],
    "LIVRE III": [
        {
            "q": "Quelles sont les mentions obligatoires d'une lettre de change pour qu'elle soit valide selon le Code de Commerce marocain ?",
            "hint_articles": [],
        },
        {
            "q": "Un chèque a été émis sans provision suffisante. Quelles sont les sanctions prévues par le Code de Commerce ?",
            "hint_articles": [],
        },
        {
            "q": "Comment fonctionne l'endossement d'une lettre de change ? Quels effets produit-il ?",
            "hint_articles": [],
        },
    ],
    "LIVRE IV": [
        {
            "q": "Un transporteur a causé des dommages à la marchandise. Quelle est sa responsabilité selon le Code de Commerce marocain ?",
            "hint_articles": [],
        },
        {
            "q": "Quelles sont les obligations d'un commissionnaire envers son commettant selon le Code de Commerce ?",
            "hint_articles": [],
        },
    ],
    "LIVRE V": [
        {
            "q": "Une entreprise rencontre des difficultés financières. Quelle est la différence entre la procédure de sauvegarde et le redressement judiciaire selon le Code de Commerce marocain ?",
            "hint_articles": [],
        },
        {
            "q": "Quelles sont les conditions d'ouverture d'une procédure de liquidation judiciaire au Maroc ?",
            "hint_articles": [],
        },
        {
            "q": "Qu'est-ce que la période suspecte dans une procédure de redressement judiciaire ? Quels effets produit-elle sur les actes antérieurs ?",
            "hint_articles": [],
        },
        {
            "q": "Quelles sont les sanctions pénales prévues par le Code de Commerce pour les dirigeants en cas de banqueroute ?",
            "hint_articles": [],
        },
    ],
}

DEFINITION_QUESTIONS = [
    ("Qu'est-ce qu'un acte de commerce au sens du Code de Commerce marocain ?", [1, 2, 6, 7]),
    ("Qui est considéré comme commerçant selon le Code de Commerce marocain ?", [1, 6, 7]),
    ("Qu'est-ce que le fonds de commerce selon le droit commercial marocain ?", [79, 80]),
    (
        "Qu'est-ce qu'une lettre de change et quelles sont ses caractéristiques essentielles ?",
        [159, 160, 161],
    ),
    ("Qu'est-ce que le crédit-bail (leasing) selon le Code de Commerce marocain ?", [431, 432]),
    ("Quelle est la définition du registre du commerce au Maroc et à quoi sert-il ?", [27, 28, 29]),
    ("Qu'est-ce que le nantissement du fonds de commerce ?", [106, 107]),
    ("Qu'est-ce qu'un billet à ordre et en quoi diffère-t-il de la lettre de change ?", [232, 233]),
    (
        "Qu'est-ce que la procédure de sauvegarde selon le Code de Commerce marocain ?",
        [546, 547, 548],
    ),
    (
        "Qu'est-ce que la conciliation dans les procédures de difficultés de l'entreprise ?",
        [551, 552],
    ),
]


# Main generator


def generate_rule_based_dataset(articles: list[dict], art_lookup: dict) -> list[TrainingExample]:
    """Generate deterministic training examples from article content."""
    examples: list[TrainingExample] = []

    for article_index, article in enumerate(articles):
        art_num = article["article_number"]
        text = article["text"]

        if not text.strip():
            continue

        # Pre-build the standard answer once — reused across question types
        a = make_direct_answer(article)

        # 1. Direct question — every article
        template = random.choice(DIRECT_TEMPLATES)
        q = template.format(n=art_num)
        examples.append(
            TrainingExample(
                instruction=q,
                input_context="",
                output=a,
                article_refs=[art_num],
                category="direct",
                source="rule_based",
            )
        )

        # 2. Semantic question — every 4th article
        # Teaches conceptual reasoning, not just article lookup
        if article_index % 4 == 0:
            template = random.choice(SEMANTIC_TEMPLATES)
            q = template.format(n=art_num)
            examples.append(
                TrainingExample(
                    instruction=q,
                    input_context="",
                    output=a,
                    article_refs=[art_num],
                    category="direct",
                    source="rule_based",
                )
            )

        # 3. Article lookup question — every 3rd article
        # Teaches "which article covers this topic?" retrieval
        def extract_snippet(text: str, max_len: int = 160) -> str:
            snippet = text[:max_len]

            if "." in snippet:
                snippet = snippet.rsplit(".", 1)[0] + "."

            return snippet.strip()

        if article_index % 3 == 0:
            snippet = extract_snippet(text)
            template = random.choice(ARTICLE_LOOKUP_TEMPLATES)
            q = template.format(snippet=snippet)
            examples.append(
                TrainingExample(
                    instruction=q,
                    input_context="",
                    output=make_lookup_answer(article),
                    article_refs=[art_num],
                    category="lookup",
                    source="rule_based",
                )
            )

        # 4. RAG-style question — every 2nd article
        # Trains model to answer from injected context (critical for RAG inference)
        if article_index % 2 == 0:
            template = random.choice(RAG_TEMPLATES)
            examples.append(
                TrainingExample(
                    instruction=template,
                    input_context=text,  # article text injected as context
                    output=make_rag_answer(article),
                    article_refs=[art_num],
                    category="rag",
                    source="rule_based",
                )
            )

        # 5. Livre-context question — subset with chapitre
        hier = article["hierarchy"]
        if hier.get("chapitre") and article_index % 3 == 0:
            template = random.choice(LIVRE_CONTEXT_TEMPLATES)
            q = template.format(
                n=art_num,
                livre=hier.get("livre", ""),
                chapitre=hier.get("chapitre", ""),
            )
            examples.append(
                TrainingExample(
                    instruction=q,
                    input_context="",
                    output=a,
                    article_refs=[art_num],
                    category="direct",
                    source="rule_based",
                )
            )

        # 6. Cross-reference question — articles with 2+ refs
        if article["cross_refs"] and len(article["cross_refs"]) >= 2:
            refs = article["cross_refs"][:3]
            refs_str = ", ".join(f"{r}" for r in refs)
            template = random.choice(CROSS_REF_TEMPLATES)
            q = template.format(n=art_num, refs=refs_str)
            a_cross = make_cross_ref_answer(article, art_lookup)
            examples.append(
                TrainingExample(
                    instruction=q,
                    input_context="",
                    output=a_cross,
                    article_refs=[art_num] + refs,
                    category="synthesis",
                    source="rule_based",
                )
            )

    # Definition questions
    for q_text, hint_arts in DEFINITION_QUESTIONS:
        relevant_articles = [art_lookup[str(n)] for n in hint_arts if str(n) in art_lookup]
        if not relevant_articles:
            continue
        hint_arts = [art["article_number"] for art in relevant_articles]
        answer_parts = ["Selon le Code de Commerce marocain (Loi n° 15-95) :\n"]
        for art in relevant_articles:
            answer_parts.append(f"**Article {art['article_number']}** : {art['text']}")
        answer_parts.append(
            "\n[Références : " + ", ".join(f"Art. {n}" for n in hint_arts) + ", Loi n° 15-95]"
        )
        examples.append(
            TrainingExample(
                instruction=q_text,
                input_context="",
                output="\n\n".join(answer_parts),
                article_refs=hint_arts,
                category="definition",
                source="rule_based",
            )
        )

    # Scenario questions
    for livre_key, scenarios in SCENARIO_TEMPLATES.items():
        for scenario in scenarios:
            examples.append(
                TrainingExample(
                    instruction=scenario["q"],
                    input_context="",
                    output=(
                        f"[Réponse générée — à enrichir avec les articles pertinents "
                        f"de {livre_key} via le pipeline de génération synthétique]"
                    ),
                    article_refs=scenario["hint_articles"],
                    category="scenario",
                    source="template_placeholder",
                )
            )

    # Out-of-scope questions — teach graceful refusal
    for q_text, domain in OUT_OF_SCOPE_QUESTIONS:
        examples.append(
            TrainingExample(
                instruction=q_text,
                input_context="",
                output=(
                    f"Cette question dépasse le périmètre du Code de Commerce marocain "
                    f"(Loi n° 15-95). Elle relève du {domain}. "
                    f"Je ne peux pas y répondre dans le cadre de cette expertise."
                ),
                article_refs=[],
                category="out_of_scope",
                source="rule_based",
            )
        )

    return examples


# Format converters


def to_chatml(example: TrainingExample) -> dict:
    """Convert to ChatML format for HuggingFace TRL SFT."""
    user_content = example.instruction
    if example.input_context:
        user_content = f"Contexte :\n{example.input_context}\n\nQuestion : {example.instruction}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.output},
        ],
        "article_refs": example.article_refs,
        "category": example.category,
        "source": example.source,
    }


def to_alpaca(example: TrainingExample) -> dict:
    """Convert to Alpaca format as alternative."""
    return {
        "instruction": example.instruction,
        "input": example.input_context,
        "output": example.output,
        "article_refs": example.article_refs,
        "category": example.category,
    }


def build_synthetic_expansion_prompt(article: dict) -> str:
    """Returns a prompt for Claude API to generate 5 additional Q&A pairs."""
    return f"""Tu es un générateur de données d'entraînement pour un modèle juridique spécialisé dans le Code de Commerce marocain.

Voici un article :
---
{article["context_header"]}
Article {article["article_number"]}
{article["text"]}
---

Génère exactement 5 questions-réponses variées et de haute qualité pour cet article.
Les questions doivent couvrir : questions directes, scénarios pratiques, définitions, comparaisons.
Chaque réponse DOIT citer l'article {article["article_number"]} et être précise juridiquement.

Réponds UNIQUEMENT en JSON valide, format :
[
  {{"question": "...", "reponse": "...", "type": "direct|scenario|definition|synthese"}},
  ...
]"""


# deduplication utility
def deduplicate_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    seen = set()
    unique = []

    for e in examples:
        key = (e.instruction, e.input_context, e.output)

        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# save
def save_dataset(examples: list[TrainingExample], output_dir: Path, articles: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out placeholder examples
    valid = [e for e in examples if e.source != "template_placeholder"]

    # ChatML JSONL — primary format for TRL SFT
    chatml_file = output_dir / "train_chatml.jsonl"
    with open(chatml_file, "w", encoding="utf-8") as f:
        for ex in valid:
            f.write(json.dumps(to_chatml(ex), ensure_ascii=False) + "\n")
    print(f"[dataset] Saved {len(valid)} ChatML examples → {chatml_file}")

    # Alpaca JSON — alternative format
    alpaca_file = output_dir / "train_alpaca.json"
    with open(alpaca_file, "w", encoding="utf-8") as f:
        json.dump([to_alpaca(e) for e in valid], f, ensure_ascii=False, indent=2)
    print(f"[dataset] Saved Alpaca format → {alpaca_file}")

    # Synthetic expansion prompts placeholder
    expansion_prompts_file = output_dir / "synthetic_expansion_prompts.jsonl"
    with open(expansion_prompts_file, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(
                json.dumps(
                    {
                        "article_number": article["article_number"],
                        "prompt": build_synthetic_expansion_prompt(article),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[dataset] Synthetic expansion prompts template → {expansion_prompts_file}")

    # Dataset composition stats
    cats = Counter(e.category for e in valid)
    sources = Counter(e.source for e in valid)
    print("\n[dataset] Examples by category:")
    for cat, count in cats.most_common():
        print(f"  {count:4d}  {cat}")
    print("\n[dataset] Examples by source:")
    for src, count in sources.most_common():
        print(f"  {count:4d}  {src}")


# Entry point
if __name__ == "__main__":
    from atlaslex.config import project_home
    from atlaslex.preparation.extraction import CDC_SHA256, document_hash, extract_pages
    from atlaslex.preparation.parsing import parse_articles

    parser = argparse.ArgumentParser(
        description="Export optional source-derived training templates (no training)"
    )
    parser.add_argument("--input", type=Path, default=project_home() / "data/raw/cdc.pdf")
    parser.add_argument("--output", type=Path, default=project_home() / "data/processed/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    digest = document_hash(args.input)
    if digest != CDC_SHA256:
        parser.error(
            "These law-specific templates are only supported for the supplied Code de commerce."
        )
    parsed, warnings = parse_articles(
        extract_pages(args.input), digest, args.input.name, specialized=True
    )
    articles = [
        {
            "article_number": a.article_number,
            "text": a.text or "\n".join(a.footnotes),
            "context_header": a.context_header,
            "hierarchy": a.hierarchy,
            "footnotes": a.footnotes,
            "cross_refs": a.cross_refs,
        }
        for a in parsed
        if a.article_number
    ]
    art_lookup = {a["article_number"]: a for a in articles}
    for article in articles:
        article["cross_refs"] = [r for r in article["cross_refs"] if r in art_lookup]
    examples = deduplicate_examples(generate_rule_based_dataset(articles, art_lookup))
    save_dataset(examples, args.output, articles)
