import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def prepare_documents(pdf_path):
    # 1. Load the PDF
    print(f"Reading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # 2. Split the PDF into small "Chunks"
    # Why? Because LLMs have a "limit" on how much they can read at once.
    # We want to give it bite-sized pieces of information.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(data)
    
    print(f"Successfully split into {len(chunks)} chunks of information.")
    return chunks

if __name__ == "__main__":
    # This is the line that actually "starts" the engine!
    # Make sure 'docs/jordan_labor_law.pdf' matches your actual file name
    prepare_documents("docs/jordan_labor_law.pdf")