#!/usr/bin/env python3
"""Step 2: map MIMIC drug names -> RxCUI ingredient -> ATC codes (RxNorm).

For every unique drug token in the drug vocab (full vocab so the random
baseline is fully covered), normalize the pharmacy string, resolve to an
RxNorm ingredient RxCUI, and fetch its ATC class codes.

Combination drugs (e.g. "carbidopa-levodopa") are split into components;
each component is mapped, and the token's ATC set is the union over
components. Each component's ingredient RxCUI(s) are kept for same-substance
(ATC level-5-equivalent) detection.

Cache: cache/drug_atc_mapping.json  -> {token_id: {...}}
Result keyed by drug token id (string).  Re-runs reuse the cache.
"""
from __future__ import annotations

import json
import re
import sys
import time

import pandas as pd
import requests

import common as C

REST = "https://rxnav.nlm.nih.gov/REST"
SESSION = requests.Session()
CACHE_FILE = C.CACHE / "drug_atc_mapping.json"
NAME_CACHE_FILE = C.CACHE / "drug_name_resolution.json"  # normalized-name -> result
SLEEP = 0.05

# pharmacy-string noise to strip
_NF = re.compile(r"\*[^*]*\*")                 # *nf*, *iv*, ...
_PAREN = re.compile(r"\([^)]*\)")              # (25-100), (iv), ...
_DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|unit|units|%|meq|iu)\b", re.I)
_TRAILNUM = re.compile(r"\b\d+(\.\d+)?\b")
# Salt/form words are NOT stripped: RxNorm's approximate matcher handles them,
# and stripping them wrongly collapses real drugs (e.g. potassium chloride).
SPLIT = re.compile(r"[-/+]| and | with ")


def normalize(raw: str) -> list[str]:
    s = raw.lower()
    s = _NF.sub(" ", s)
    s = _PAREN.sub(" ", s)
    s = _DOSE.sub(" ", s)
    parts = [p.strip() for p in SPLIT.split(s)]
    comps = []
    for p in parts:
        p = _TRAILNUM.sub(" ", p)
        p = re.sub(r"[^a-z0-9 ]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 3:
            comps.append(p)
    # dedup preserve order
    seen, out = set(), []
    for c in comps:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _get(url, params):
    for _ in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            time.sleep(0.5)
        time.sleep(0.3)
    return None


def name_to_rxcui(name: str) -> str | None:
    j = _get(f"{REST}/rxcui.json", {"name": name, "search": "2"})
    if j:
        ids = j.get("idGroup", {}).get("rxnormId")
        if ids:
            return ids[0]
    j = _get(f"{REST}/approximateTerm.json",
             {"term": name, "maxEntries": "1", "option": "1"})
    if j:
        cands = j.get("approximateGroup", {}).get("candidate")
        if cands:
            return cands[0].get("rxcui")
    return None


def rxcui_to_atc(rxcui: str) -> list[str]:
    j = _get(f"{REST}/rxclass/class/byRxcui.json",
             {"rxcui": rxcui, "relaSource": "ATC"})
    if not j:
        return []
    info = j.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
    codes = set()
    for it in info:
        ci = it.get("rxclassMinConceptItem", {})
        if it.get("relaSource") == "ATC" and ci.get("classId"):
            codes.add(ci["classId"])
    return sorted(codes)


def resolve_component(name: str, name_cache: dict) -> dict:
    if name in name_cache:
        return name_cache[name]
    rxcui = name_to_rxcui(name)
    time.sleep(SLEEP)
    atc = rxcui_to_atc(rxcui) if rxcui else []
    time.sleep(SLEEP)
    res = {"name": name, "rxcui": rxcui, "atc": atc}
    name_cache[name] = res
    return res


def main():
    drug_map = pd.read_csv(C.DRUG_MAP)
    id2name = dict(zip(drug_map["drug_id"].astype(str),
                       drug_map["drug_name"].astype(str)))
    vocab = pd.read_csv(C.OUT / "vocab_drug.csv")["id"].astype(str).tolist()

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    name_cache = (json.loads(NAME_CACHE_FILE.read_text())
                  if NAME_CACHE_FILE.exists() else {})

    todo = [t for t in vocab if t not in cache]
    print(f"drug vocab={len(vocab)}  cached={len(vocab)-len(todo)}  to map={len(todo)}")

    for i, tid in enumerate(todo, 1):
        raw = id2name.get(tid, tid)
        comps = normalize(raw)
        comp_results = [resolve_component(c, name_cache) for c in comps]
        atc = sorted({code for cr in comp_results for code in cr["atc"]})
        rxcuis = [cr["rxcui"] for cr in comp_results if cr["rxcui"]]
        cache[tid] = {
            "raw": raw,
            "components": comps,
            "rxcuis": rxcuis,
            "atc": atc,
            "mapped": bool(atc),
        }
        if i % 25 == 0 or i == len(todo):
            CACHE_FILE.write_text(json.dumps(cache, indent=0))
            NAME_CACHE_FILE.write_text(json.dumps(name_cache, indent=0))
            sys.stdout.write(f"\r  mapped {i}/{len(todo)}")
            sys.stdout.flush()
    print()

    CACHE_FILE.write_text(json.dumps(cache, indent=0))
    NAME_CACHE_FILE.write_text(json.dumps(name_cache, indent=0))

    n_mapped = sum(1 for t in vocab if cache.get(t, {}).get("mapped"))
    in_pairs = pd.read_csv(C.OUT / "thesaurus_pairs.csv")
    pair_tokens = set(in_pairs[in_pairs.token_type == "drug"]["id_a"].astype(str)) | \
        set(in_pairs[in_pairs.token_type == "drug"]["id_b"].astype(str))
    n_pair_mapped = sum(1 for t in pair_tokens if cache.get(t, {}).get("mapped"))
    print(f"\nVocab mapped to ATC: {n_mapped}/{len(vocab)} "
          f"({100*n_mapped/len(vocab):.1f}%)")
    print(f"Thesaurus drug tokens mapped: {n_pair_mapped}/{len(pair_tokens)} "
          f"({100*n_pair_mapped/len(pair_tokens):.1f}%)")
    print(f"Cache -> {CACHE_FILE}")


if __name__ == "__main__":
    main()
