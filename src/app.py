import time
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup the Environment
load_dotenv()
st.set_page_config(page_title="Jordan Labor Law AI", page_icon="⚖️")

st.title("⚖️ Jordan Labor Law Assistant")
st.markdown("Ask any question about employment, leaves, or contracts in Jordan.")

# 2. Load the "Brain"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_db = Chroma(
    persist_directory="./vector_db", 
    embedding_function=embeddings
)

# 3. Setup the "Voice" (Updated for 2026 Stability)
# We are switching to gemini-2.0-flash which is the current stable workhorse
# 3. Setup the "Voice" (The "Forever Stable" Alias)
# Using 'gemini-flash-latest' ensures you always use the current working version
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

# 4. The "Special Instruction" (System Prompt)
template = """
You are a professional legal assistant specializing in the Jordanian Labor Law. 
Use the following pieces of retrieved context to answer the user's question. 

Instructions:
- If the answer is found in the context, provide a detailed response.
- If the context mentions the relevant Article (like Article 61) but doesn't show the full detail, summarize what is available and suggest the user check that specific Article.
- Always respond in the same language the user used (Arabic or English).

Context: {context}
Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. Create the MODERN Search-and-Answer Chain (LCEL)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This creates a "Pipe" line: Context -> Prompt -> Model -> Text
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | QA_CHAIN_PROMPT
    | llm
    | StrOutputParser()
)

# 6. The User Interface
# 1. Create the variable by defining the input box
user_query = st.text_input("Enter your question (Arabic or English):")

# 2. Now you can use it!
if user_query:
    with st.spinner("Searching the Labor Law..."):
        response = rag_chain.invoke(user_query)
        st.write("### Answer:")
        st.write(response)