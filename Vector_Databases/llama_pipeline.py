from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import os

os.environ['HF_TOKEN'] = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Sentiment analysis pipeline

analyzer = pipeline("sentiment-analysis")

# Question answering pipeline, specifying the checkpoint identifier

oracle = pipeline(

    "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"

)

print('Oracle ', oracle)

# Named entity recognition pipeline, passing in a specific model and tokenizer

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

#print('Model ', model)

tokeniser = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

print('Tokeniser ', tokeniser)

recogniser = pipeline("ner", model=model, tokenizer=tokeniser)

print('Recogniser ', recogniser)

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
