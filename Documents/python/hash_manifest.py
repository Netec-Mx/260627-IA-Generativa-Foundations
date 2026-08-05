import hashlib
import json
import os

hashes = {}

# Obtener ruta dinámica del usuario actual
user_downloads = os.path.join(os.environ["USERPROFILE"], "Downloads", "brochures")

for file in os.listdir(user_downloads):
    path = os.path.join(user_downloads, file)
    with open(path, "rb") as f:
        content = f.read()
        hash_value = hashlib.sha256(content).hexdigest()
        hashes[file] = hash_value

with open("hash_manifest.json", "w") as f:
    json.dump(hashes, f, indent=2)