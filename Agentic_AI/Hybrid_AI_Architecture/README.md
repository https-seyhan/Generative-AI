# fincrime_llm_ml

Local (offline-capable after model download) 3-class financial crime surveillance system:
- Local LLM via Ollama (feature extraction)
- Sentence-transformer embeddings
- LightGBM multiclass classifier

## Setup

1) Install Ollama and download a model:
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve
   ollama pull llama3

2) Python environment:
   pip install -r requirements.txt

3) Build features:
   cd src
   python build_features.py

4) Train model:
   python train_model.py

5) Predict:
   python predict_case.py

Classes:
0 = Normal
1 = Suspicious
2 = Likely Financial Crime
