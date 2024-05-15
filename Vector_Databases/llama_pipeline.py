from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import os

os.environ['HF_TOKEN'] = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Sentiment analysis pipeline

analyzer = pipeline("sentiment-analysis")

# Question answering pipeline, specifying the checkpoint identifier

oracle = pipeline(

    "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"

)
