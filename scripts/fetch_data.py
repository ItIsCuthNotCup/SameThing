"""Download the public Leipzig DBS entity-resolution benchmarks and record provenance (URL, date, sha256)."""
import datetime as dt
import hashlib
import json
import os
import sys
import zipfile

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PAGE = "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution"
SOURCES = {
    "abt-buy": "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip",
    "amazon-google": "https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(only=None):
    os.makedirs(RAW, exist_ok=True)
    prov_path = os.path.join(RAW, "PROVENANCE.json")
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {"benchmark_page": PAGE, "datasets": {}}
    for name, url in SOURCES.items():
        if only and name not in only:
            continue
        zpath = os.path.join(RAW, os.path.basename(url))
        out = os.path.join(RAW, name)
        if not os.path.exists(zpath):
            print("downloading", url)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            open(zpath, "wb").write(r.content)
            prov["datasets"].setdefault(name, {})["download_date_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if not os.path.isdir(out):
            with zipfile.ZipFile(zpath) as z:
                z.extractall(out)
        d = prov["datasets"].setdefault(name, {})
        d.update({"url": url, "sha256": sha256(zpath), "files": sorted(os.listdir(out)),
                  "license_note": "Research benchmark published by the Database Group, University of Leipzig (Köpcke, Thor, Rahm, VLDB 2010). "
                                  "Used here for evaluation only; product data is historical (c. 2010) and prices are not current offers."})
        d.setdefault("download_date_utc", dt.datetime.now(dt.timezone.utc).isoformat())
    json.dump(prov, open(prov_path, "w"), indent=2)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
