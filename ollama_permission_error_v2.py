Perfect 👍 — here’s a **ready-made Databricks init script** that ensures Ollama always runs on the driver, bound to `0.0.0.0:10134` so your notebook can reach it without permission issues.

---

## 📝 Init Script (DBFS)

Run this once in a Databricks notebook to create the init script in DBFS:

```python
script = """#!/bin/bash
set -eux

# Install Ollama binary if not present
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Only start Ollama on the driver
if [[ "$DB_IS_DRIVER" == "true" ]]; then
  # Kill any old instance
  pkill -f "ollama serve" || true

  # Set safe host/port
  export OLLAMA_HOST=0.0.0.0
  export OLLAMA_PORT=10134

  # Start in background
  nohup ollama serve > /var/log/ollama.log 2>&1 &
fi
"""

path = "dbfs:/databricks/init/ollama/ollama-driver.sh"
dbutils.fs.mkdirs("dbfs:/databricks/init/ollama")
dbutils.fs.put(path, script, overwrite=True)
print(f"Init script written to {path}")
```

---

## ⚙️ Cluster Setup

1. Go to **Cluster → Advanced Options → Init Scripts**
2. Add:

   ```
   dbfs:/databricks/init/ollama/ollama-driver.sh
   ```
3. Restart the cluster.

---

## 📓 Notebook Usage

After restart, point your Python client to the driver’s Ollama:

```python
import os, ollama

# Point to the correct host/port
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:10134"

resp = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role":"user","content":"Hello from Databricks"}]
)
print(resp["message"]["content"])
```

---

## 🔍 Verify Service is Up

```python
import urllib.request, json
url = "http://127.0.0.1:10134/api/version"
print(json.loads(urllib.request.urlopen(url).read()))
```

You should see something like:

```json
{"version": "0.1.x"}
```

---

✅ This ensures:

* Ollama runs **only on the driver**
* Bound to `0.0.0.0:10134` so notebook sees it
* Still private (not exposed outside driver VM)

---

Do you want me to also give you the **offline-model staging version** of this init script (so it doesn’t call `ollama pull` at all, just loads your pre-packaged models)?
