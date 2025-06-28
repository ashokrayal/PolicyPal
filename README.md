# PolicyPal - Enterprise RAG Chatbot

## 🎯 Project Overview

**PolicyPal** is a production-grade conversational AI chatbot that answers employee queries by retrieving answers from unstructured policy documents using Retrieval-Augmented Generation (RAG) techniques.

### Key Features
- 📄 **Multi-format Document Support**: PDF, DOCX, CSV, and OCR for scanned documents
- 🔍 **Hybrid Search**: Combines semantic (FAISS) and keyword (BM25) search
- 💬 **Conversational AI**: Context-aware responses with memory
- 🛡️ **Safety & Moderation**: Content filtering and sensitive information protection
- 📊 **Evaluation & Analytics**: Comprehensive metrics and user feedback
- 🚀 **Production Ready**: Docker deployment and monitoring

### Technology Stack
- **Python 3.10+**
- **LangChain** - RAG framework
- **FAISS** - Vector similarity search
- **SentenceTransformers** - Text embeddings
- **Streamlit** - User interface
- **FastAPI** - REST API (optional)
- **Docker** - Containerization

---

## 📋 Requirements

### System Requirements
- Python 3.10 or higher
- 4GB+ RAM (8GB+ recommended)
- 2GB+ disk space for models and data

### Dependencies
All dependencies are listed in `requirements.txt` and include:
- Core RAG: LangChain, FAISS, SentenceTransformers
- Document Processing: PyMuPDF, python-docx, pdfplumber
- UI/API: Streamlit, FastAPI, uvicorn
- Utilities: pandas, numpy, pyyaml, loguru

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd PolicyPal
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the Application
```bash
# Copy and edit configuration
cp config/config.example.yaml config/config.yaml
```

### 5. Run the Application
```bash
# Start Streamlit UI
streamlit run ui/streamlit_app.py

# Or start FastAPI backend (optional)
uvicorn api.main:app --reload
```

---

## 📁 Project Structure

```
PolicyPal/
├── src/                    # Core application code
│   ├── ingestion/         # Document processing
│   ├── embeddings/        # Vector embeddings
│   ├── retrieval/         # Search and retrieval
│   ├── conversation/      # Chat and memory
│   ├── safety/           # Content moderation
│   ├── evaluation/       # Metrics and testing
│   └── utils/            # Utilities and helpers
├── ui/                    # User interface
│   ├── streamlit_app.py  # Main Streamlit app
│   └── components/       # UI components
├── api/                   # REST API (optional)
│   ├── main.py          # FastAPI application
│   └── endpoints/       # API endpoints
├── data/                  # Data storage
│   ├── documents/       # Input documents
│   ├── embeddings/      # Generated embeddings
│   └── logs/            # Application logs
├── tests/                 # Test suite
├── docs/                  # Documentation
├── config/                # Configuration files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)
```yaml
embedding:
  model: "all-MiniLM-L6-v2"
  batch_size: 32
  max_length: 512

retrieval:
  faiss_weight: 0.7
  bm25_weight: 0.3
  top_k: 5

chunking:
  chunk_size: 1000
  chunk_overlap: 200

safety:
  enable_moderation: true
  sensitive_patterns: []
  fallback_message: "I couldn't find relevant information for your query."

logging:
  level: "INFO"
  file_path: "data/logs/policypal.log"
```

### Environment Variables
Create a `.env` file for sensitive configuration:
```env
# API Keys (if using external models)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database (if using external storage)
DATABASE_URL=your_database_url

# Security
SECRET_KEY=your_secret_key
```

---

## 📖 Usage

### Document Ingestion
1. Place your policy documents in `data/documents/`
2. Supported formats: PDF, DOCX, CSV
3. Run the ingestion pipeline:
   ```python
   from src.ingestion.pipeline import DocumentIngestionPipeline
   
   pipeline = DocumentIngestionPipeline()
   pipeline.process_documents("data/documents/")
   ```

### Chat Interface
1. Start the Streamlit app: `streamlit run ui/streamlit_app.py`
2. Upload documents or use existing ones
3. Ask questions about your policies
4. View source citations and provide feedback

### API Usage
```python
import requests

# Send a query
response = requests.post("http://localhost:8000/chat", json={
    "query": "What is the vacation policy?",
    "conversation_id": "user123"
})

print(response.json())
```

---

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_ingestion.py
```

### Evaluation
```bash
# Run evaluation on test queries
python src/evaluation/evaluation_framework.py
```

---

## 📊 Performance Metrics

### Target Metrics
- **Retrieval Quality**: Recall@5 > 0.8, Precision@5 > 0.7
- **Response Time**: < 3 seconds for typical queries
- **User Satisfaction**: > 4.0/5.0 average rating
- **Safety**: 100% detection of sensitive content patterns

### Monitoring
- Application logs: `data/logs/policypal.log`
- Performance metrics: Available in Streamlit dashboard
- User feedback: Stored in `data/feedback/`

---

## 🐳 Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t policypal .

# Run with Docker Compose
docker-compose up -d

# Or run standalone
docker run -p 8501:8501 policypal
```

### Docker Compose Configuration
```yaml
version: '3.8'
services:
  policypal:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - LOG_LEVEL=INFO
```

---

## 🔒 Security & Safety

### Content Moderation
- Regex-based filtering for sensitive patterns
- Input/output validation
- Sensitive information redaction
- Fallback responses for inappropriate queries

### Data Privacy
- Local processing (no data sent to external services by default)
- Configurable data retention policies
- Secure logging practices

---

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Commit with clear messages: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Create a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for all classes and methods
- Include unit tests for new features

---

## 📚 Documentation

### Additional Documentation
- [Technical Architecture](docs/technical_architecture.md)
- [API Documentation](docs/api_documentation.md)
- [User Manual](docs/user_manual.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Evaluation Report](docs/evaluation_report.md)

### Learning Resources
- [RAG Concepts](docs/rag_concepts.md)
- [System Design](docs/system_design.md)
- [Interview Q&A](docs/interview_qa.md)

---

## 🐛 Troubleshooting

### Common Issues

**1. Memory Issues**
```bash
# Increase Python memory limit
export PYTHONMALLOC=malloc
```

**2. FAISS Installation**
```bash
# Install CPU version
pip install faiss-cpu

# Or GPU version (if available)
pip install faiss-gpu
```

**3. OCR Issues**
```bash
# Install Tesseract
# On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# On macOS: brew install tesseract
# On Ubuntu: sudo apt-get install tesseract-ocr
```

### Getting Help
- Check the [logs](data/logs/) for error details
- Review the [troubleshooting guide](docs/troubleshooting.md)
- Open an issue on GitHub

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** team for the excellent RAG framework
- **FAISS** team for efficient similarity search
- **SentenceTransformers** for high-quality embeddings
- **Streamlit** for the beautiful UI framework

---

## 📞 Contact

- **Project**: PolicyPal Enterprise RAG Chatbot
- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [your-github-profile]

---

*Built with ❤️ for enterprise policy management* 