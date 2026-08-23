#!/bin/bash
# RAGORA başlatma scripti
# Demo/sunum öncesi bu script ile başlat — 27b modeli için yeterli RAM sağlar

cd "$(dirname "$0")"

# Ollama'yı yeniden başlat (VRAM ve sistem belleğini temizler)
echo "[1/3] Ollama yeniden başlatılıyor..."
pkill -f ollama 2>/dev/null
sleep 2
ollama serve &>/tmp/ollama.log &
sleep 3

# 27b modeli aktif et
echo "[2/3] Model ayarlanıyor (27b)..."
sed -i 's/^#OLLAMA_MODEL=qwen3.5:27b/OLLAMA_MODEL=qwen3.5:27b/' .env
sed -i 's/^OLLAMA_MODEL=qwen3.5:9b/#OLLAMA_MODEL=qwen3.5:9b/' .env

# Streamlit başlat
echo "[3/3] RAGORA başlatılıyor..."
pkill -f "streamlit run" 2>/dev/null
sleep 1
streamlit run streamlit_app.py --server.port 8501

