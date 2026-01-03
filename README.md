# 📘 Multilingual NCERT Doubt Solver using RAG + LoRA Fine-Tuned LLM

This project is a **Retrieval-Augmented Generation (RAG)** based NCERT doubt-solving system that answers questions directly from NCERT textbooks.

It uses:
- A **LoRA fine-tuned TinyLlama model** (trained by me)
- **FAISS vector search** over NCERT PDFs
- **LangChain-based RAG pipeline**
- Fully **local CPU inference**

⚠️ The system is intentionally CPU-based and slow to demonstrate **model training + inference**, not speed.

---

## 🧠 Problem Statement

**Multilingual NCERT Doubt-Solver using OPEA-based RAG Pipeline**

Students often struggle to get reliable answers grounded in NCERT textbooks.  
This system ensures:
- Answers are **strictly based on NCERT content**
- No hallucinations
- Transparent retrieval + generation

---

## 🏗️ Architecture Overview

1. **Document Ingestion**
   - NCERT PDFs (Classes 5–10)
   - Loaded using `PyPDFLoader`

2. **Text Chunking**
   - RecursiveCharacterTextSplitter
   - Small chunks for better retrieval

3. **Embedding Generation**
   - Sentence Transformers (`all-MiniLM-L6-v2`)

4. **Vector Store**
   - FAISS for fast similarity search
   - Index built once and reused

5. **LLM**
   - Base: TinyLlama-1.1B
   - Fine-tuned using **LoRA**
   - Loaded locally using Hugging Face + PEFT

6. **RAG Pipeline**
   - Retriever → Prompt → LLM → Clean Answer

---

## 🚀 How to Run the Project (Step-by-Step)

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd ncert_rag
