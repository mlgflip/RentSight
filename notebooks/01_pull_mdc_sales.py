"""Pull Miami-Dade Property Appraiser parcel + sale records (single-family & condo)
from the county ArcGIS REST service into data/raw/mdc_sales_raw.csv.

Source: MD_ComparableSales/MapServer/5 (MDC.PaGis) on gisweb.miamidade.gov
"""
import csv
import json
import time
import urllib.parse
import urllib.request

BASE = "https://gisweb.miamidade.gov/arcgis/rest/services/MD_ComparableSales/MapServer/5/query"
WHERE = (
    "PRICE_1>25000 AND BEDROOM_COUNT>0 AND BUILDING_HEATED_AREA>200 "
    "AND DOS_1>='20200101' AND (DOR_CODE_CUR='0101' OR CONDO_FLAG='Y')"
)
FIELDS = [
    "FOLIO", "TRUE_SITE_ZIP_CODE", "TRUE_SITE_CITY", "DOR_CODE_CUR", "DOR_DESC",
    "CONDO_FLAG", "BEDROOM_COUNT", "BATHROOM_COUNT", "HALF_BATHROOM_COUNT",
    "FLOOR_COUNT", "BUILDING_HEATED_AREA", "BUILDING_ACTUAL_AREA", "LOT_SIZE",
    "YEAR_BUILT", "DOS_1", "PRICE_1", "DOS_2", "PRICE_2",
]
PAGE = 20000
OUT = "data/raw/mdc_sales_raw.csv"


def fetch(offset):
    params = {
        "where": WHERE,
        "outFields": ",".join(FIELDS),
        "returnGeometry": "true",
        "outSR": "4326",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE),
        "orderByFields": "FOLIO",
        "f": "json",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}; retrying in 15s")
            time.sleep(15)
    raise RuntimeError(f"giving up at offset {offset}")


def main():
    offset, total = 0, 0
    with open(OUT, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS + ["LON", "LAT"])
        while True:
            d = fetch(offset)
            feats = d.get("features", [])
            if not feats:
                break
            for f in feats:
                a = f["attributes"]
                g = f.get("geometry") or {}
                writer.writerow([a.get(k) for k in FIELDS] + [g.get("x"), g.get("y")])
            total += len(feats)
            print(f"offset {offset}: +{len(feats)} rows (total {total})", flush=True)
            if not d.get("exceededTransferLimit") and len(feats) < PAGE:
                break
            offset += PAGE
    print(f"DONE: {total} rows -> {OUT}")


if __name__ == "__main__":
    main()
