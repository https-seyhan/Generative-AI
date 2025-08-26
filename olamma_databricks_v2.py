Perfect ✅ — if you want **zero internet** dependency and **repeatable cluster spin-ups**, the cleanest way is to bake a **custom Docker image** that already contains:

* The **Ollama binary**
* Your chosen **models** (weights + manifests)

Then you point your Databricks cluster at that image.

Here’s how to do it step by step:

---

# 🐳 1. Build a Custom Ollama Image (locally or in CI/CD)

On a machine with internet:

```dockerfile
# Dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y curl bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy pre-downloaded models into the image
# (from ~/.ollama/models on your local machine)
COPY models /root/.ollama/models

# Expose Ollama API port
EXPOSE 11434

# Start Ollama server by default
CMD ["ollama", "serve"]
```

### Build the image:

```bash
# First, export your local models from ~/.ollama
tar -czf ollama-models.tar.gz -C ~/.ollama models
mkdir models && tar -xzf ollama-models.tar.gz -C models

# Build Docker image
docker build -t my-ollama:offline .
```

At this point `my-ollama:offline` contains **Ollama + your models**.

---

# 📦 2. Push to a Private Registry

Push the image to whatever registry Databricks can pull from:

```bash
# Example: AWS ECR
aws ecr create-repository --repository-name my-ollama
docker tag my-ollama:offline <your-account>.dkr.ecr.<region>.amazonaws.com/my-ollama:offline
docker push <your-account>.dkr.ecr.<region>.amazonaws.com/my-ollama:offline
```

Databricks also supports Azure Container Registry (ACR) or Google Artifact Registry — same idea.

---

# ⚙️ 3. Configure Databricks Cluster to Use Custom Image

In **Databricks Cluster UI**:

* Runtime: **Databricks Container Services** (must be enabled for custom Docker).
* Under **Advanced Options → Docker**:

  * Image URL:
    `<your-registry-url>/my-ollama:offline`
  * Add credentials if registry is private.

This way every node boots with Ollama + your models already inside.

---

# 🚀 4. Use Ollama from Notebook (No Init Scripts Needed)

Once the cluster starts, you already have the Ollama server running inside the container.
Just install the Python client:

```python
%pip install --quiet ollama
dbutils.library.restartPython()
```

And run as before:

```python
import ollama

resp = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role":"user", "content":"Explain Databricks in 12 words"}]
)
print(resp["message"]["content"])
```

No `ollama pull`, no internet, 100% local.

---

# 🔑 Why This is Stronger than Init Scripts

* **Self-contained**: image already has binary + weights.
* **Fast boot**: no downloads or unpacking.
* **Reproducible**: versioned image → same environment every time.
* **Offline secure**: works in air-gapped VPCs.

---

👉 Question for you: do you want this Docker image to run **Ollama server only on the driver** (for notebooks), or do you also want **executors/workers to run their own Ollama instances** for parallel inference?
