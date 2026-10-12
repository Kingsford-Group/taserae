# Similarity vs performance

Bins TASER-AE synthetic records by their similarity to the nearest real record and
trains one classifier per bin, three seeds.

```bash
python3 revision/similarity_performance/run_full_pipeline.py --device cuda:0
```
Run from the repository root. The pipeline calls `compute_similarity_scores.py`,
`build_bin_datasets.py`, `run_bin_benchmark.py` and `analyze_similarity_performance.py`
in turn. Results go to `revision/similarity_performance/outputs/`.
