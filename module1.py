from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

import PyPDF2
import pandas as pd
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ===== Step 1: Load and Extract Text =====
def load_pdf_text(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_csv_text(file_path):
    df = pd.read_csv(file_path)
    return "\n".join([f"Row {i}: {row.to_dict()}" for i, row in df.iterrows()])

# ===== Step 2: Create Embeddings =====
def create_vectorstore_from_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# ===== Step 3: Prompt Template & Model =====
template = """
Answer the question below using only the provided context.

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
llm = OllamaLLM(model="llama3")
chain = prompt | llm

# ===== Step 4: Run Chatbot =====
def handle_conversation(retriever):
    print("ChatBot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        docs = retriever.invoke(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        for chunk in chain.stream({"context": context, "question": user_input}):
            print(chunk, end="", flush=True)
        print()

# ===== Step 5: Main Logic =====
def main():
    file_path = input("Enter full path to PDF or CSV: ").strip()
    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:
        if file_path.endswith(".pdf"):
            text = load_pdf_text(file_path)
        elif file_path.endswith(".csv"):
            text = load_csv_text(file_path)
        else:
            print("Unsupported file format. Only PDF and CSV are supported.")
            return

        if not text.strip():
            print("No readable text found in the file.")
            return

        print("Processing document...")
        vectorstore = create_vectorstore_from_text(text)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        handle_conversation(retriever)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()