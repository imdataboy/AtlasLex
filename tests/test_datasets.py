import runpy
from pathlib import Path

from atlaslex.preparation.parsing import article_references

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_optional_scripts_import_without_anthropic():
    runpy.run_path(str(EXAMPLES / "generate_dataset.py"))
    runpy.run_path(str(EXAMPLES / "legacy_synthetic.py"))


def test_article_groups_do_not_leak_and_split_is_repeatable():
    functions = runpy.run_path(str(EXAMPLES / "legacy_synthetic.py"))
    examples = [
        {"article_refs": refs, "messages": []}
        for refs in [[1], [2], [1, 2], [3], [4], [3, 4], [5], [6], [7], [8], []]
    ]
    train, evaluation = functions["split_examples"](examples, seed=42)
    assert train and evaluation
    assert not (
        {r for e in train for r in e["article_refs"]}
        & {r for e in evaluation for r in e["article_refs"]}
    )
    assert (train, evaluation) == functions["split_examples"](examples, seed=42)


def test_rule_based_context_is_not_truncated(tmp_path):
    module = runpy.run_path(str(EXAMPLES / "generate_dataset.py"))
    article = {
        "article_number": "1",
        "text": "Texte juridique. " * 200,
        "context_header": "LIVRE PREMIER",
        "hierarchy": {},
        "cross_refs": [],
        "footnotes": [],
    }
    examples = module["generate_rule_based_dataset"]([article], {"1": article})
    rag = next(e for e in examples if e.category == "rag")
    assert rag.input_context == article["text"]
    module["save_dataset"](examples, tmp_path, [article])
    assert (tmp_path / "synthetic_expansion_prompts.jsonl").stat().st_size > 0


def test_abbreviated_citations_are_detected():
    assert article_references("Art. 999 et 1000") == ["1000", "999"]
