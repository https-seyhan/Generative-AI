#Since Databricks doesn’t distinguish driver vs worker when pulling custom Docker image, the trick is:

* Ship Ollama + models in the image.
* Auto-start `ollama serve` only **if the node is the driver**.

Databricks sets env vars you can use (`DB_IS_DRIVER=true` on the driver node).

```dockerfile
FROM ubuntu:22.04

# Install deps
RUN apt-get update && apt-get install -y curl bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy pre-baked models
COPY models /root/.ollama/models

# Run Ollama only if this is the driver
CMD if [ "$DB_IS_DRIVER" = "true" ]; then \
      ollama serve; \
    else \
      tail -f /dev/null; \
    fi
```

* **Driver node** runs Ollama automatically.
* **Worker nodes** do nothing (just idle container).

---

# Cluster Config in Databricks

When creating cluster:

* Runtime: **Databricks Container Services**.
* Docker image: `<your-registry-url>/my-ollama:driver-only`.
* Make sure driver has enough RAM/CPU (or GPU if you use GPU models).

Workers don’t run Ollama, so they just do Spark work.

---

#  Notebook Usage (driver-side only)

Install the Python client in your notebook:

```python
%pip install --quiet ollama
dbutils.library.restartPython()
```

After restart, you can chat with your local model:

```python
import ollama

resp = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role":"user","content":"Give me a 1-line summary of Delta Lake"}]
)
print(resp["message"]["content"])
```

---

# 🗂 4. Batch Workflow Pattern

Since only the **driver** can talk to Ollama, keep inference driver-side.
That means:

```python
import pandas as pd

questions = [
    "What is MLflow?",
    "Write a haiku about Databricks."
]

df = pd.DataFrame({"q": questions})

def ask(q):
    return ollama.chat(model="llama3.1:8b", messages=[{"role":"user","content":q}])["message"]["content"]

df["a"] = [ask(q) for q in df.q]
df
```

⚠️ Don’t call from Spark workers (`mapPartitions`) — they won’t see Ollama.
If you need large-scale inference, coalesce to 1 partition and run on driver, or expose the driver port inside your VPC (less private, but possible).

---

# Benefits of Driver-only Setup

* Saves cluster resources (no wasted workers running Ollama).
* Keeps LLM inference private and local.
* Simple to manage (no networking between nodes).

---

Do you want me to also show you how to **enable GPU acceleration on the driver** (so Ollama uses CUDA when available), or are you planning to stick with CPU inference?
