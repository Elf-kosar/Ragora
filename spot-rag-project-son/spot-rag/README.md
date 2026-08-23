# Spot RAG Pipeline

Boston Dynamics Spot kullanım kılavuzu için **Docling** tabanlı,
context-preserving PDF → RAG sistemi.

**Stack:** Docling · MongoDB · Qdrant · sentence-transformers · BGE

---

## Mimari

```
PDF
 │
 ▼
DoclingPDFParser          ← layout-aware, tablo+başlık+breadcrumb korumalı
 │
 ▼
SemanticChunker           ← yapısal bölümleme + cümle sınırı overlap
 │
 ▼
LocalEmbedder             ← GPU'da BAAI/bge-large (1024 dim)
 │
 ├──► MongoDB              ← ham metin + metadata (full-text search)
 └──► Qdrant               ← vektörler (semantic search)

Query
 │
 ├── embed_query()
 ├── Qdrant semantic search
 ├── MongoDB keyword search   (hybrid)
 └── RRF fusion → top-k chunks → LLM context
```

---

## Kurulum

### 1. Servisleri başlat

```bash
docker compose up -d
```

### 2. Python ortamı

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ayarlar

```bash
cp .env.example .env
# .env dosyasını düzenle (gerekirse model değiştir)
```

### 4. PDF'i kopyala

```bash
mkdir data
cp /path/to/spot-user-manual-en.pdf data/
```

---

## İndeksleme

```bash
# İlk çalıştırma
python -m pipeline.indexer --pdf data/spot-user-manual-en.pdf

# Sıfırdan yeniden indeksle
python -m pipeline.indexer --pdf data/spot-user-manual-en.pdf --reset
```

Beklenen çıktı (GPU'ya göre değişir):
```
ADIM 1/4: PDF Parse      → ~850 eleman
ADIM 2/4: Chunking       → ~420 chunk  
ADIM 3/4: Embedding      → ~2 dakika (RTX 3090)
ADIM 4/4: Store          → MongoDB + Qdrant
İNDEKSLEME TAMAMLANDI  — ~180s
```

---

## Chatbot

```bash
python chatbot.py
```

`chatbot.py` içindeki `call_llm()` fonksiyonunu kendi LLM'inle değiştir:

### Ollama
```python
import requests
def call_llm(system, user):
    r = requests.post("http://localhost:11434/api/generate",
        json={"model": "llama3.1", "system": system,
              "prompt": user, "stream": False})
    return r.json()["response"]
```

### vLLM / LM Studio (OpenAI-compatible)
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
def call_llm(system, user):
    r = client.chat.completions.create(
        model="your-model",
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}])
    return r.choices[0].message.content
```

---

## Retriever API

```python
from pipeline.retriever import Retriever

retriever = Retriever(top_k=5, use_hybrid=True)

# Sorgula
chunks = retriever.retrieve("How do I perform an emergency stop?")

# Context string oluştur
context = retriever.format_context(chunks)

# Sadece tablo ara
table_chunks = retriever.retrieve(
    "payload weight limits",
    filter_chunk_type="table"
)
```

---

## Model Seçimi

| Model | Dim | Dil | Hız | Kalite |
|-------|-----|-----|-----|--------|
| `BAAI/bge-large-en-v1.5` | 1024 | EN | ★★★ | ★★★★★ |
| `BAAI/bge-base-en-v1.5` | 768 | EN | ★★★★ | ★★★★ |
| `intfloat/multilingual-e5-large` | 1024 | TR+EN | ★★★ | ★★★★ |
| `BAAI/bge-m3` | 1024 | Çok dilli | ★★ | ★★★★★ |

`.env` dosyasında `EMBED_MODEL` değişkenini değiştir.

---

## Proje Yapısı

```
spot-rag/
├── parsers/
│   └── docling_parser.py     # PDF → ParsedElement
├── chunkers/
│   └── semantic_chunker.py   # ParsedElement → Chunk
├── embedders/
│   └── local_embedder.py     # Chunk text → float32 vector
├── stores/
│   ├── mongo_store.py        # Chunk metadata store
│   └── qdrant_store.py       # Vector store
├── pipeline/
│   ├── indexer.py            # İndeksleme pipeline
│   └── retriever.py          # Retrieval + hybrid search
├── config/
│   └── settings.py           # Merkezi ayarlar
├── chatbot.py                # CLI demo
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
