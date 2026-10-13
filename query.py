import os
import sys

print("[DEBUG] Script started...")

DB_DIR = "./chroma_db"

def query_database(query_text):
    try:
        if not os.path.exists(DB_DIR):
            print(f"[ERROR] Database directory '{DB_DIR}' not found. Please run 'ingest.py' first.")
            return

        print(f"[1/2] Loading local embedding model (all-MiniLM-L6-v2)...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print(f"[1.5/2] Connecting to ChromaDB at '{DB_DIR}'...")
        from langchain_community.vectorstores import Chroma
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        print(f"[2/2] Searching for: '{query_text}'...\n")
        results = vector_store.similarity_search_with_score(query_text, k=3)

        if not results:
            print("[INFO] No matching documents found in the database.")
            return

        print("=" * 60)
        print("RETRIEVED CONTEXT CHUNKS:")
        print("=" * 60)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[Match {i}] (Distance Score: {score:.4f})")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content:\n{doc.page_content}")
            print("-" * 60)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your search query about the document: ")

    
    query_database(user_query)