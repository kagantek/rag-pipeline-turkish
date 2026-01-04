# Turkish Enterprise RAG Pipeline

A high-precision AI RAG assistant for Turkish enterprise documentation. Processes ERP manuals, annual reports, and tax legislation with strict adherence to numerical facts.

## Features

- **Turkish-Native**: Optimized for Turkish agglutinative language
- **Zero-Hallucination**: Strict system prompt prevents fabricated numbers
- **Two-Stage Retrieval**: Vector search + FlashRank reranking
- **Model Flexibility**: Hot-swap between Llama 3.3 70B, Llama 3.1 8B

## Installation

### 1. Clone and Setup Environment

```bash
cd rag-pipeline-turkish
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
API key from Groq, or any other LLM provider must be put in the .env file.

## Usage

### Start the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:{PORT}`

### Workflow

1. **Upload PDFs**: Use the sidebar to upload financial documents
2. **Process Documents**: Click "Dökümanları İşle" to index
3. **Select Model**: Choose LLM (Llama 3.3 70B for accuracy, Gemma 2 9B for speed)
4. **Enable Reranking**: Toggle for higher precision (recommended)
5. **Ask Questions**: Type your query in Turkish

### Example Questions

- "KDV oranı nedir?"
- "Fatura kesim tarihi ne zaman?"
- "Amortisman süresi kaç yıl?"
- "Logo Tiger'da stok kartı nasıl açılır?"


## Configuration

Edit `config.py` to customize chunks size, top k chunks etc.

## Technical Details

### Models Used

Embeddings: `intfloat/multilingual-e5-small`
Reranker: `ms-marco-MiniLM-L-12-v2`
LLM: Groq Cloud (Llama - 70b - 8b)

### E5 Model Prefixing (Critical)

The multilingual-e5 model requires specific prefixes:
- **Documents**: `passage: {text}` 
- **Queries**: `query: {text}`

This is automatically handled by the `embeddings.py` module.
