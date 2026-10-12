# Computational cost

Wall-clock time of embedding generation, TASER-AE augmentation, classifier training
and test inference.

```bash
python3 revision/computational_cost/run_full_pipeline_cost.py --run_embeddings --device cuda:0
```
Run from the repository root. The `measure_*.py` scripts time each stage on its own.
