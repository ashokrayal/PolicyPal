# Search Pipeline API Documentation

This document describes the API for the unified search pipeline integrating semantic (FAISS), keyword (BM25), and hybrid retrieval.

---

## Methods / Endpoints

### 1. Add Documents
- **Method:** `add_documents(documents: List[Dict], text_key: str = "content")`
- **Description:** Add and index documents in the pipeline.
- **Parameters:**
  - `documents`: List of dicts, each with at least a text field and metadata.
  - `text_key`: Key for text content in each document (default: "content").
- **Example:**
```python
pipeline.add_documents([
    {"content": "Company policy on leave.", "file_name": "policy1.pdf", "category": "HR"},
    {"content": "Remote work guidelines.", "file_name": "policy2.pdf", "category": "HR"}
])
```

---

### 2. Semantic Search (FAISS)
- **Method:** `semantic_search(query: str, top_k: int = 5, metadata_filter: dict = None)`
- **Description:** Perform semantic search using vector similarity.
- **Parameters:**
  - `query`: Search query string.
  - `top_k`: Number of top results to return.
  - `metadata_filter`: Optional dict to filter results by metadata fields.
- **Example:**
```python
results = pipeline.semantic_search("leave policy", top_k=3, metadata_filter={"category": "HR"})
```
- **Response Format:**
```json
[
  {
    "content": "Company policy on leave.",
    "file_name": "policy1.pdf",
    "category": "HR",
    "score": 0.123
  },
  ...
]
```

---

### 3. Keyword Search (BM25)
- **Method:** `keyword_search(query: str, top_k: int = 5)`
- **Description:** Perform keyword-based search using BM25.
- **Parameters:**
  - `query`: Search query string.
  - `top_k`: Number of top results to return.
- **Example:**
```python
results = pipeline.keyword_search("remote work", top_k=2)
```
- **Response Format:**
```json
[
  {
    "content": "Remote work guidelines.",
    "file_name": "policy2.pdf",
    "category": "HR",
    "score": 7.12
  },
  ...
]
```

---

### 4. Hybrid Search
- **Method:** `hybrid_search(query: str, top_k: int = 5, metadata_filter: dict = None)`
- **Description:** Perform hybrid search combining semantic and keyword results.
- **Parameters:**
  - `query`: Search query string.
  - `top_k`: Number of top results to return.
  - `metadata_filter`: Optional dict to filter results by metadata fields.
- **Example:**
```python
results = pipeline.hybrid_search("leave policy", top_k=5, metadata_filter={"category": "HR"})
```
- **Response Format:**
```json
[
  {
    "content": "Company policy on leave.",
    "file_name": "policy1.pdf",
    "category": "HR",
    "hybrid_score": 0.87,
    "faiss_score": 0.12,
    "bm25_score": 7.1
  },
  ...
]
```

---

### 5. Index Management
- **Method:** `save_indexes()`
  - **Description:** Save FAISS index and metadata to disk.
- **Method:** `load_indexes()`
  - **Description:** Load FAISS index and metadata from disk.

---

## Notes
- All search methods return a list of dicts, each containing the document content, metadata, and scores.
- Metadata filtering is supported for semantic and hybrid search.
- The pipeline can be used as a backend for a REST API, FastAPI, or Streamlit app. 