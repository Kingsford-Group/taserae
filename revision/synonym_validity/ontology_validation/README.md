# Ontology validation of synonym pairs

Checks thesaurus pairs against the ATC (via RxNorm) and ICD-10-CM hierarchies,
compared with a random-pair baseline. Run the steps in order:

```bash
cd revision/synonym_validity/ontology_validation
python3 build_pairs.py
python3 map_drugs_to_atc.py           # queries the public NLM RxNav API
python3 map_complications_to_icd.py   # queries the public NLM Clinical Tables API
python3 compute_agreement.py
python3 generate_latex_tables.py
```
Needs `data/drug_id_map.csv` and `data/complication_id_map.csv`, written by
`data/mimic_all_ade.py`. API responses are cached in `cache/`; results go to `outputs/`.
