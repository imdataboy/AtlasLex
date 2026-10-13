# Optional utilities and historical assets

The main application is `uv run atlaslex`. These utilities are separate and do
not run during indexing or chat.

```bash
# Export rule-based ChatML, Alpaca, and expansion prompts from the supplied PDF.
uv run python examples/generate_dataset.py

# Deterministically merge existing datasets and split connected article groups.
uv run python examples/legacy_synthetic.py --merge-only
```

Outputs go to `data/processed/datasets/`. The examples are templated source
excerpts; their legal reasoning quality has not been evaluated. No fine-tuning
is implemented. The unfinished Claude API generation path is explicitly disabled;
its prompt templates remain available for historical reference. Anthropic is not
required. The merge utility accepts an existing `synthetic_chatml.jsonl` if supplied.
Training and evaluation partitions never share article references, though their
sizes need not be exactly 95/5 because related articles stay together.

`fill_db.py` and `rrv.py` retain the original manual Qdrant toy examples, using a
separate `test_collection`. Run the fill example only when that collection does
not already exist. They perform no operations on import.

`legacy_data/` retains the original generated artifacts locally, marked as
historical and Git-ignored. The application never indexes them. The sanitized
original notebook is at `notebooks/original_rag.ipynb` and is not a supported
execution path.
