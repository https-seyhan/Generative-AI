import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
import math

# ----------------------------
# 1. Load models
# ----------------------------
# LLM
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Embedding model (for retrieval)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 2. Knowledge Base
# ----------------------------
documents = [
    "This property has a large backyard suitable for children.",
    "The house is close to schools and public transport.",
    "A family-friendly neighbourhood with many parks nearby.",
    "This apartment does not have outdoor space.",
    "The villa comes with a private garden and swimming pool."
]

doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# ----------------------------
# 3. Tools
# ----------------------------
def retrieve(query, top_k=2):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [documents[i] for i in indices[0]]

def summarise(text):
    prompt = f"Summarise the following text in 2 sentences:\n{text}\nSummary:"
    return generate_text(prompt, max_length=80)

def calculator(expression):
    try:
        result = eval(expression, {"__builtins__": None, "math": math}, {})
        return str(result)
    except Exception:
        return "Error: invalid expression"

# ----------------------------
# 4. LLM generation
# ----------------------------
def generate_text(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# 5. Agentic Multi-Tool Loop
# ----------------------------
def agentic_rag(user_query, max_steps=4):
    reasoning_log = []
    context = ""

    for step in range(max_steps):
        # Ask model which tool to use
        control_prompt = f"""
You are an assistant with access to tools.
Available tools:
- RETRIEVE[query]: search documents
- SUMMARISE[text]: condense info
- CALCULATE[expression]: evaluate math
- FINAL[answer]: produce final answer

User asked: "{user_query}"
Current context: "{context}"

Decide the next action. Use exactly one tool in the format TOOLNAME[input].
"""
        action = generate_text(control_prompt, max_length=100)
        reasoning_log.append(f"[Step {step+1} Action] {action}")

        # Parse tool and input
        if "RETRIEVE[" in action.upper():
            query = action.split("[",1)[1].rsplit("]",1)[0]
            docs = retrieve(query)
            context += "\n".join(docs) + "\n"
            reasoning_log.append(f"[Step {step+1} Retrieval] {docs}")

        elif "SUMMARISE[" in action.upper():
            text = action.split("[",1)[1].rsplit("]",1)[0]
            summary = summarise(text)
            context += summary + "\n"
            reasoning_log.append(f"[Step {step+1} Summarise] {summary}")

        elif "CALCULATE[" in action.upper():
            expr = action.split("[",1)[1].rsplit("]",1)[0]
            result = calculator(expr)
            context += f"Calculation result: {result}\n"
            reasoning_log.append(f"[Step {step+1} Calculator] {expr} = {result}")

        elif "FINAL[" in action.upper():
            final_answer = action.split("[",1)[1].rsplit("]",1)[0]
            reasoning_log.append(f"[Final Answer] {final_answer}")
            return final_answer, reasoning_log

        else:
            reasoning_log.append(f"[Step {step+1}] Unrecognised action: {action}")
            break

    # Fallback if loop ended without FINAL
    fallback_prompt = f"Based on context:\n{context}\n\nAnswer the question:\n{user_query}"
    final_answer = generate_text(fallback_prompt, max_length=200)
    reasoning_log.append(f"[Final Answer - Fallback] {final_answer}")
    return final_answer, reasoning_log

# ----------------------------
# 6. Run Example
# ----------------------------
user_query = "I'm looking for a family-friendly home with a backyard. Do you have any properties like that?"
answer, log = agentic_rag(user_query)

print("Generated Answer:\n", answer)
print("\nReasoning Log:")
for entry in log:
    print(entry)

# ----------------------------
# 7. Save to File
# ----------------------------
output_file = "generated_output.txt"
with open(output_file, "a", encoding="utf-8") as f:
    f.write("\n" + "="*50 + "\n")
    f.write("Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    f.write("User Query:\n" + user_query + "\n\n")
    f.write("Generated Answer:\n" + answer + "\n\n")
    f.write("Reasoning Log:\n" + "\n".join(log) + "\n")

print(f"\nOutput appended to {output_file}")
