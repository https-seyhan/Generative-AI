# Azure OpenAI Regression Framework

## Overview
Enterprise-grade regression testing framework for Azure OpenAI with:
- PySpark distributed evaluation
- Prompt Flow / Azure ML integration
- LangGraph-style orchestration
- CI/CD ready

## Run Locally
```bash
pip install -r requirements.txt
python jobs/run_regression.py
```

## Run PySpark
```bash
spark-submit jobs/run_regression_spark.py
```
