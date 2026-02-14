Minimal scheduler service

Create:

scheduler/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
import requests
import logging

DATABASE_URL = "postgresql://risk:risk@postgres:5432/riskdb"

jobstores = {
    'default': SQLAlchemyJobStore(url=DATABASE_URL)
}

scheduler = BackgroundScheduler(jobstores=jobstores)

# ---- JOBS ----

def recompute_features():
    requests.post("http://feature_engine:8001/rebuild")

def retrain_model():
    requests.post("http://ml_model:8002/retrain")

def daily_case_review():
    requests.post("http://policy_engine:8005/review_open_cases")

def drift_monitor():
    requests.post("http://ml_model:8002/drift")

# ---- SCHEDULES ----

# every 15 minutes
scheduler.add_job(recompute_features, 'interval', minutes=15)

# nightly retraining
scheduler.add_job(retrain_model, 'cron', hour=2, minute=0)

# daily compliance review
scheduler.add_job(daily_case_review, 'cron', hour=4)

# hourly monitoring
scheduler.add_job(drift_monitor, 'interval', hours=1)

scheduler.start()

print("Risk scheduler started")

import time
while True:
    time.sleep(60)

Dockerfile for scheduler

scheduler/Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scheduler.py .

CMD ["python", "scheduler.py"]


Add to docker-compose.yml:

  risk_scheduler:
    build: ./scheduler
    depends_on:
      - postgres
      - feature_engine
      - ml_model
      - policy_engine

What jobs you should schedule (very important)
1) Behavioural Feature Windows

Fraud and phoenixing are temporal behaviours.

You must recompute rolling windows:

1 hour

24 hour

7 day

30 day

This cannot be event-triggered because inactivity is itself a signal.

2) Concept Drift Detection

Your LightGBM will silently decay.

Schedule:

PSI (Population Stability Index)
KS test
prediction distribution shift


If PSI > 0.25 → alert.

3) Automatic Escalation Aging

Compliance requirement:

If a suspicious case sits unresolved:

Age	Action
24h	reminder
72h	escalate
7d	compliance breach flag

APScheduler enforces this.

4) LLM Memory Compression

LLM investigator logs grow fast.

Nightly job:

summarise past investigations

store embeddings

delete raw prompts

This drastically reduces token costs.

Why not just use cron?

Cron cannot:

call internal containers easily

store job state

handle retries

be container aware

access DB sessions

APScheduler can.

It becomes your system’s temporal brain — handling processes that depend on time, not events.
