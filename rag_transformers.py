"""
RAG pipeline using LoRA-fine-tuned TinyLlama model.
This version demonstrates model training + inference.
Inference is CPU-based and intentionally slow.
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


# =========================
# LOAD MODEL (BASE + LoRA)
# =========================
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_ID = "Akankshamulik/tinyllama-lora-training"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, ADAPTER_ID)


# =========================
# LLM GENERATION FUNCTION
# =========================
def llm_generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# FORMAT DOCUMENTS
# =========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =========================
# LOAD & SPLIT PDF
# =========================
loader = PyPDFLoader("data/kech106.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)


# =========================
# EMBEDDINGS + VECTOR STORE
# =========================
import os

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if not os.path.exists("faiss_index"):
    print("🔧 Building FAISS index (one-time)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
else:
    print("⚡ Loading FAISS index...")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# LIMIT retrieval to top 2 (FASTER + CLEANER)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# =========================
# PROMPT
# =========================
prompt = PromptTemplate.from_template(
    """Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:"""
)


# =========================
# RAG CHAIN (CORRECT)
# =========================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | RunnableLambda(lambda p: p.to_string())
    | RunnableLambda(llm_generate)
    | StrOutputParser()
)


# =========================
# RUN QUERY
# =========================
print("\n✅ RAG system ready")
print("Type your question below (type 'exit' to quit)\n")

while True:
    question = input("🧠 Question: ")

    if question.lower() in ["exit", "quit"]:
        print("\n👋 Exiting RAG system. Bye!")
        break

    answer = rag_chain.invoke(question)

    print("\n🔹 Answer:\n")
    print(answer)
    print("\n" + "-" * 60 + "\n")
