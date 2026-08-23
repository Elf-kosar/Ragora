"""
Merkezi ayar yönetimi — tüm parametreler tek yerden.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB
    mongo_uri: str = Field("mongodb://localhost:27017", env="MONGO_URI")
    mongo_db: str = Field("spot_rag", env="MONGO_DB")
    mongo_collection: str = Field("chunks", env="MONGO_COLLECTION")

    # Qdrant
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_collection: str = Field("spot_chunks", env="QDRANT_COLLECTION")

    # Embedding
    embed_model: str = Field("intfloat/multilingual-e5-large", env="EMBED_MODEL")
    embed_batch_size: int = Field(32, env="EMBED_BATCH_SIZE")
    embed_device: str = Field("cuda", env="EMBED_DEVICE")

    # Chunking
    chunk_size: int = Field(1500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(64, env="MIN_CHUNK_SIZE")

    # CLIP (görsel embedding)
    clip_model: str = Field("openai/clip-vit-large-patch14", env="CLIP_MODEL")

    # Visual store
    visual_mongo_collection: str = Field("visual_chunks", env="VISUAL_MONGO_COLLECTION")
    visual_qdrant_collection: str = Field("spot_visual_chunks", env="VISUAL_QDRANT_COLLECTION")

    # VLM
    vlm_model: str = Field("llava-hf/llava-v1.6-34b-hf", env="VLM_MODEL")

    # Ollama
    ollama_host: str = Field("http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field("qwen2.5:32b", env="OLLAMA_MODEL")
    ollama_vision_model: str = Field("llava:34b", env="OLLAMA_VISION_MODEL")

    # Pipeline
    pdf_path: Path = Field(Path("./data/spot-user-manual-en.pdf"), env="PDF_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
