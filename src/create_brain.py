import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from ingest_data import prepare_documents 

load_dotenv()

def build_vector_store():
    pdf_path = "docs/jordan_labor_law.pdf"
    chunks = prepare_documents(pdf_path)

    print("🧠 Initializing Gemini Embedding Model...")
    # This is the most stable model name for 2026
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        task_type="retrieval_document"
    )

    # TINY BATCHES for the Free Tier
    batch_size = 10 
    vector_db = None

    print(f"🚀 Starting process for {len(chunks)} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch_num = (i // batch_size) + 1
        print(f"⏳ Processing batch {current_batch_num} of {len(chunks)//batch_size + 1}...")

        try:
            if vector_db is None:
                vector_db = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory="./vector_db"
                )
            else:
                vector_db.add_documents(batch)
            
            print(f"✅ Batch {current_batch_num} finished.")
            
            # THE "FREELANCER PAUSE"
            # We wait 60 seconds after EVERY 10 chunks to ensure the quota resets
            print("😴 Sleeping for 60 seconds to stay under the Free Tier limit...")
            time.sleep(60)

        except Exception as e:
            print(f"⚠️ Error in batch {current_batch_num}: {e}")
            print("🛑 Waiting 90 seconds before trying the next batch...")
            time.sleep(90) # Extra long sleep if we hit a limit

    print("🏁 FINAL SUCCESS! The 'vector_db' folder is complete.")

if __name__ == "__main__":
    build_vector_store()