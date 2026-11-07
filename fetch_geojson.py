"""
Run this once to download and fix the Tunisia governorates GeoJSON.
Needs internet only for this one-time run (project itself stays offline).
Usage: python fetch_geojson.py
"""
import json
import urllib.request

SOURCE_URL = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/644497a5c35bfd834152fb4e378834ad910ee327/releaseData/gbOpen/TUN/ADM1/geoBoundaries-TUN-ADM1.geojson"

NAME_MAP = {
    "Tunis": "Tunis",
    "Ben Arous": "Ben Arous",
    "El Kef": "Le Kef",
    "Sousse": "Sousse",
    "Sfax": "Sfax",
    "Jendouba": "Jendouba",
    "Kairouan": "Kairouan",
    "Kasserine": "Kasserine",
    "Mahdia": "Mahdia",
    "Manouba": "Mannouba",
    "Sidi Bouzid": "Sidi Bou Zid",
    "Kébili": "Kebili",
    "Béja": "Beja",
    "Tataouine": "Tataouine",
    "Gabès": "Gabes",
    "Bizerte": "Bizerte",
    "Ariana": "Ariana",
    "Nabeul": "Nabeul",
    "Monastir": "Monastir",
    "Siliana": "Siliana",
    "Zaghouan": "Zaghouan",
    "Gafsa": "Gafsa",
    "Médenine": "Mednine",
    "Tozeur": "Tozeur",
}

print("Downloading source GeoJSON...")
with urllib.request.urlopen(SOURCE_URL) as resp:
    data = json.loads(resp.read().decode("utf-8"))

out_features = []
for feat in data["features"]:
    shape_name = feat["properties"]["shapeName"]
    gouv_fr = NAME_MAP[shape_name]
    gouv_id = "TN-" + feat["properties"]["shapeISO"]
    out_features.append({
        "type": "Feature",
        "properties": {"gouv_id": gouv_id, "gouv_fr": gouv_fr},
        "geometry": feat["geometry"],
    })

out = {"type": "FeatureCollection", "features": out_features}

with open("tunisia_governorates.geojson", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"Done. Wrote {len(out_features)} governorates to tunisia_governorates.geojson")
print("Now move this file into your project's assets/ folder, replacing the old one.")