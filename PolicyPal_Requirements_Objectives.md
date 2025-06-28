# Project: PolicyPal – An Enterprise RAG Chatbot

## 🎯 Objective Document

### Project Name:
**PolicyPal: A Production-Grade Conversational AI using Retrieval-Augmented Generation (RAG)**

### Core Objective:
To design and implement an enterprise-grade chatbot that answers employee queries by retrieving answers from unstructured policy documents using RAG techniques. The solution will include ingestion, preprocessing, vector and keyword-based retrieval, conversational context, evaluation, UI, safety, and deployment.

### Learning-Driven Sub-Objective:
The project is intentionally structured to help developers **gradually learn RAG concepts and system design principles**, with increasing complexity. It begins with basic document handling and builds up to advanced hybrid search, moderation, evaluation, and deployment practices.

### Project Goals:
- ✅ Build a working Conversational AI chatbot for internal policy documents.
- ✅ Ingest and preprocess documents in real-world formats (PDF, DOCX, CSV).
- ✅ Implement semantic and keyword-based hybrid search (FAISS + BM25).
- ✅ Integrate with LangChain’s ConversationalRetrievalChain.
- ✅ Support fallback logic, moderation filters, and user feedback.
- ✅ Provide a user-facing interface (Streamlit) and optionally FastAPI.
- ✅ Deploy the solution via Docker with support for future scalability.
- ✅ Build deep technical understanding of GenAI RAG workflows.

---

## 📋 Requirements Document

### Functional Requirements

1. **Document Ingestion and Preprocessing**
   - Parse PDFs (layout-aware), DOCX, CSV.
   - Clean and normalize text.
   - Handle OCR-based scanned PDFs.
   - Chunk content with recursive splitter.
   - Add metadata (source, category, page).

2. **Embedding & Vector Store**
   - Generate embeddings using SentenceTransformers (e.g., MiniLM).
   - Store embeddings in FAISS index.
   - Save aligned metadata for each chunk.

3. **Keyword Search (BM25)**
   - Create a BM25 retriever using LangChain or LlamaIndex.
   - Normalize and tokenize input.

4. **Hybrid Retrieval**
   - Use LangChain EnsembleRetriever.
   - Tune weights between FAISS and BM25.
   - Add metadata filtering (by category/source).

5. **Conversational Pipeline**
   - Integrate LangChain ConversationalRetrievalChain.
   - Add memory and prompt templates.
   - Cite sources in the response.

6. **Fallbacks and Moderation**
   - Return fallback message when no relevant chunks.
   - Add regex filters or OpenAI moderation for safety.
   - Support redaction of sensitive inputs.

7. **Feedback and Logging**
   - Log each query, response, retrieved chunks, and scores.
   - Allow thumbs up/down input from user.
   - Save logs and feedback to file or CSV.

8. **Evaluation**
   - Track retrieval quality (Recall@k).
   - Run manual evaluation on 10+ queries.
   - Optionally calculate BLEU/ROUGE scores.

9. **UI/Interface**
   - Build a chatbot UI using Streamlit.
   - Optional: expose an API using FastAPI.

10. **Deployment**
    - Write Dockerfile and docker-compose config.
    - Enable local or cloud deployment.
    - Log startup state and usage stats.

---

### Non-Functional Requirements

- Code must be modular and documented.
- All intermediate outputs should be saved for traceability.
- Git should be used for version control with daily commits.
- Code should support extension to LangSmith, Pinecone, Redis.
- Streamlit UI should be responsive and minimal.

---

### Tools and Libraries

- Python 3.10+
- LangChain
- FAISS
- SentenceTransformers (`all-MiniLM-L6-v2`)
- BM25 (via LangChain or LlamaIndex)
- Streamlit
- FastAPI (optional)
- Docker
- PyMuPDF, pdfplumber, python-docx
- pdf2image + pytesseract (for OCR)
- Pandas
- Git

---

### Deliverables

- Complete `policy-pal/` project repository
- Chunked and embedded document store
- Working chatbot with conversational memory
- Log files and feedback summary CSV
- Streamlit app or API server
- Docker image and deployment instructions
- Capstone demo queries + evaluation writeup
- Interview prep summary PDF (10+ Q&A)

---

This objective and requirements document defines the foundation for building, learning from, and demonstrating an advanced RAG chatbot project aligned with real-world engineering goals.
