#!/usr/bin/env python3
"""Step 3: map complication descriptions -> ICD-10-CM codes.

Complication tokens are SNOMED-style text descriptions with no codes in the
source map, so we resolve each description to its best ICD-10-CM code via the
free NLM Clinical Tables API (sf=code,name searches the description field).

We keep the top hit and a difflib confidence = string similarity between the
query and the returned ICD description; low-confidence matches are flagged so
agreement can be reported with/without a confidence floor.

Threaded (NLM allows concurrent requests). Cache: cache/complication_icd_mapping.json
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import pandas as pd
import requests

import common as C

API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
CACHE_FILE = C.CACHE / "complication_icd_mapping.json"
WORKERS = 12
MAXLIST = 25      # candidates to fetch, then re-rank locally
CONF_FLOOR = 0.5  # informational flag threshold (token recall)

_lock = threading.Lock()
_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


_STOP = {"of", "the", "and", "with", "without", "in", "on", "to", "a", "an",
         "due", "or", "by", "for", "unspecified", "other"}


def clean(name: str) -> str:
    s = re.sub(r"[\"']", " ", str(name))
    s = re.sub(r"[+]", " and ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


def _score(qtoks: list[str], desc: str) -> float:
    """Rank candidate ICD description against the query.

    Primary: recall of (content) query tokens. Bonus if the description
    begins with the query phrase (favours the generic 'X, unspecified' code
    over an obscure qualified variant). Small penalty per extra word.
    """
    if not qtoks:
        return 0.0
    dtoks = _tokens(desc)
    dset = set(dtoks)
    recall = sum(t in dset for t in qtoks) / len(qtoks)
    prefix = 0.3 if " ".join(dtoks[: len(qtoks)]) == " ".join(qtoks) else 0.0
    extra = 0.01 * max(0, len(dtoks) - len(qtoks))
    return recall + prefix - extra


def _api(q: str) -> list[list[str]]:
    for _ in range(3):
        try:
            r = session().get(
                API,
                params={"sf": "code,name", "terms": q, "maxList": str(MAXLIST)},
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                return data[3] if data[0] and len(data) > 3 else []
        except requests.RequestException:
            continue
    return []


def query_icd(name: str) -> dict:
    q = clean(name)
    qtoks = _tokens(q)
    disp = _api(q)
    used_q = q
    # fallback: drop trailing content words until the API returns candidates
    drop = list(qtoks)
    while not disp and len(drop) > 1:
        drop = drop[:-1]
        used_q = " ".join(drop)
        disp = _api(used_q)
    if not disp:
        return {"query": q, "icd": None, "mapped": False, "n_hits": 0}

    best = max(disp, key=lambda cd: _score(qtoks, cd[1]))
    code, desc = best[0], best[1]
    recall = (sum(t in set(_tokens(desc)) for t in qtoks) / len(qtoks)
              if qtoks else 0.0)
    return {
        "query": q,
        "matched_query": used_q,
        "icd": code,
        "icd_desc": desc,
        "confidence": round(recall, 3),
        "str_sim": round(SequenceMatcher(None, q.lower(), desc.lower()).ratio(), 3),
        "n_hits": len(disp),
        "relaxed": used_q != q,
        "mapped": True,
    }


def main():
    comp_map = pd.read_csv(C.COMP_MAP)
    id2name = dict(zip(comp_map["complication_id"].astype(str),
                       comp_map["complication_name"].astype(str)))
    vocab = pd.read_csv(C.OUT / "vocab_complication.csv")["id"].astype(str).tolist()

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    todo = [t for t in vocab if t not in cache]
    print(f"complication vocab={len(vocab)}  cached={len(vocab)-len(todo)}  "
          f"to map={len(todo)}", flush=True)

    done = [0]

    def work(tid):
        res = query_icd(id2name.get(tid, tid))
        with _lock:
            cache[tid] = res
            done[0] += 1
            if done[0] % 100 == 0:
                CACHE_FILE.write_text(json.dumps(cache, indent=0))
                sys.stdout.write(f"\r  mapped {done[0]}/{len(todo)}")
                sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo))
    CACHE_FILE.write_text(json.dumps(cache, indent=0))
    print()

    n_map = sum(1 for t in vocab if cache.get(t, {}).get("mapped"))
    n_conf = sum(1 for t in vocab
                 if cache.get(t, {}).get("mapped")
                 and cache[t].get("confidence", 0) >= CONF_FLOOR)
    pairs = pd.read_csv(C.OUT / "thesaurus_pairs.csv")
    ct = pairs[pairs.token_type == "complication"]
    ptok = set(ct["id_a"].astype(str)) | set(ct["id_b"].astype(str))
    n_pt = sum(1 for t in ptok if cache.get(t, {}).get("mapped"))
    print(f"Vocab mapped to ICD-10: {n_map}/{len(vocab)} ({100*n_map/len(vocab):.1f}%)")
    print(f"  of which confidence>={CONF_FLOOR}: {n_conf} ({100*n_conf/len(vocab):.1f}%)")
    print(f"Thesaurus complication tokens mapped: {n_pt}/{len(ptok)} "
          f"({100*n_pt/len(ptok):.1f}%)")
    print(f"Cache -> {CACHE_FILE}")


if __name__ == "__main__":
    main()
