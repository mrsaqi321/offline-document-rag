import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DB_DIR = "./chroma_db"

def run_rag(question: str):
    if not os.path.exists(DB_DIR):
        print(f"[ERROR] Database directory '{DB_DIR}' not found. Please run 'ingest.py' first.")
        return

    print("[1/4] Loading local embedding model and retrieving context...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings
    )
    
    # Perform similarity search directly
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    print("[2/4] Loading local Qwen2.5-0.5B model into CPU memory...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32  # Standard precision safe for CPU
    )

    print("[3/4] Constructing grounded prompt...")
    prompt = f"""You are a technical AI assistant. Answer the question accurately using ONLY the provided context. If the answer is not in the context, say "Information not found."

Context:
{context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt")

    print("[4/4] Generating response locally on CPU...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode output tokens back into text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up output to isolate the model's answer
    if "Answer:" in full_output:
        answer = full_output.split("Answer:")[-1].strip()
    else:
        answer = full_output.strip()

    print("=" * 60)
    print("100% LOCAL NATIVE RAG RESPONSE:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your question for the local RAG system: ")
    
    run_rag(user_query)