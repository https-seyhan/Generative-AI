from transformers import BertTokenizer, BertForSequenceClassification # for phrase matching
from datasets import load_dataset

# Load a dataset for phrase matching (e.g., STS-B)
dataset = load_dataset("glue", "stsb")

# Load tokenizer and BERT model for sequence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)  # For regression task

def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

# Apply the tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)

# Initialize the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()
# Evaluate the model on the validation set
trainer.evaluate()

# Example pair of phrases
phrase1 = "How to fine-tune BERT for phrase matching?"
phrase2 = "How can I fine-tune BERT for matching phrases?"

# Tokenize the input phrases
inputs = tokenizer(phrase1, phrase2, return_tensors='pt', truncation=True, padding=True)

# Get the model's prediction
outputs = model(**inputs)
prediction = torch.sigmoid(outputs.logits)

print(f"Similarity score: {prediction.item()}")

model.save_pretrained('./fine_tuned_bert_phrase_matching')
tokenizer.save_pretrained('./fine_tuned_bert_phrase_matching')

