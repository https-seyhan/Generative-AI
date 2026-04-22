azure-openai-regression-framework/
│
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── configs/
│   ├── model_config.yml
│   ├── evaluation_config.yml
│   ├── safety_config.yml
│
├── data/
│   ├── golden_dataset/
│   │   ├── v1/
│   │   │   ├── classification.json
│   │   │   ├── summarization.json
│   │   │   └── extraction.json
│   │   └── v2/
│   ├── test_runs/
│   │   ├── 2026-04-22/
│   │   └── history.parquet
│
├── src/
│   ├── core/
│   │   ├── client.py
│   │   ├── prompt_runner.py
│   │   └── embeddings.py
│   │
│   ├── evaluation/
│   │   ├── semantic.py
│   │   ├── llm_judge.py
│   │   ├── rules.py
│   │   └── scorer.py
│   │
│   ├── safety/
│   │   ├── content_safety.py
│   │   └── thresholds.py
│   │
│   ├── pipelines/
│   │   ├── regression_pipeline.py
│   │   └── batch_runner.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── config_loader.py
│   │   └── schema_validator.py
│
├── tests/
│   ├── test_semantic.py
│   ├── test_rules.py
│   └── test_pipeline.py
│
├── notebooks/
│   ├── exploratory_eval.ipynb
│
├── dashboards/
│   └── streamlit_app.py
│
├── jobs/
│   ├── run_regression.py
│   └── backfill_runs.py
│
├── ci/
│   └── bitbucket-pipelines.yml
│
└── outputs/
    ├── reports/
    └── logs/
