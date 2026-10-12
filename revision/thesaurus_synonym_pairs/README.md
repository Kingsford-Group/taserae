# Thesaurus synonym pairs

Samples synonym pairs from the Word2Vec thesaurus by cosine-similarity range, with
drug and complication names, for qualitative inspection.

```bash
python3 revision/thesaurus_synonym_pairs/sample_cosine_synonym_pairs.py
```
Run from the repository root. Needs `data/drug_id_map.csv` and `data/complication_id_map.csv`,
written by `data/mimic_all_ade.py`.
