from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
import torch
import json


[
  {
    "query": "What are the key tasks in piling bored activities?",
    "documents": ["Document 1: Drilling description...", "Document 2: Reinforcement and concrete pouring..."],
    "response": "The key tasks in piling bored activities include drilling, reinforcement, and concrete pouring."
  },
  {
    "query": "How to optimize task scheduling for foundation work?",
    "documents": ["Document 1: Best practices for foundation...", "Document 2: Scheduling optimization techniques..."],
    "response": "To optimize task scheduling for foundation work, prioritize excavation and follow-up with concurrent activities like reinforcement and pouring."
  }
]


# Load custom construction dataset
def load_custom_construction_dataset(path):
    with open(path, "r") as f:
        dataset = json.load(f)
    return dataset

construction_dataset = load_custom_construction_dataset("construction_schedule_dataset.json")

# Initialize RAG tokenizer and retriever
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact")

# Initialize the RAG model with T5 generator
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Define optimizer
optimizer = torch.optim.AdamW(rag_model.parameters(), lr=5e-5)

# Fine-tune on construction dataset
rag_model.train()

# Training loop
for epoch in range(3):  # You can adjust the number of epochs
    for sample in construction_dataset:
        query = sample['query']
        documents = sample['documents']
        response = sample['response']
        
        # Tokenize input and retrieved documents
        input_ids = tokenizer(query, return_tensors="pt", padding=True).input_ids
        doc_input_ids = tokenizer(documents, return_tensors="pt", padding=True).input_ids
        
        # Tokenize response as labels
        labels = tokenizer(response, return_tensors="pt", padding=True).input_ids

        # Forward pass
        outputs = rag_model(input_ids=input_ids, doc_input_ids=doc_input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the fine-tuned model
rag_model.save_pretrained("fine-tuned-rag-construction")
