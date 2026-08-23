"""
RAG (Retrieval-Augmented Generation) Pipeline
MongoDB (metadata) + Qdrant (vectors) + Llama ile TÃ¼rkÃ§e Soru-Cevap Sistemi
"""
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from bson import ObjectId

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
    SOURCE_DISPLAY_THRESHOLD,
    MAX_CONTEXT_CHARS,
    CATEGORIES,
    TURKISH_QA_PROMPT,
)
from database.mongodb_manager import MongoDBManager
from database.qdrant_manager import QdrantManager
from rag.context_safety import build_untrusted_context_block, sanitize_retrieved_text
from rag.model_utils import strip_thinking, with_no_think


class RAGSystem:
    """RAG tabanlÄ± soru-cevap sistemi"""

    CATEGORY_ALIASES = {
        "warthog": ["warthog"],
        "jackal": ["jackal"],
        "dingo": ["dingo"],
        "boxer": ["boxer"],
        "ridgeback": ["ridgeback"],
        "husky_a200": ["husky a200", "a200"],
        "husky_a300": ["husky a300", "a300"],
        "husky_a300_amp": ["husky a300 amp", "a300 amp"],
        "husky_a300_observer": ["husky a300 observer", "a300 observer"],
    }

    QUERY_INTENT_ALIASES = {
        "speed": [
            "hiz", "hizi", "hizli", "speed", "velocity", "maksimum hiz",
            "max speed", "maximum speed", "top speed", "en yuksek hiz",
            "gidebilir", "ulasabilecegi",
        ],
        "payload": [
            "payload", "yuk", "tasima", "tasiyabilir", "kapasite",
            "load", "maximum payload",
        ],
        "weight": ["agirlik", "agirligi", "kutle", "mass", "weight", "kg"],
        "runtime": [
            "calisma suresi", "runtime", "run time", "battery life",
            "pil suresi", "batarya suresi", "ne kadar calisir",
        ],
        "battery": [
            "batarya", "battery", "sarj", "charge", "charger", "voltaj",
            "voltage", "amper", "current",
        ],
        "dimensions": [
            "boyut", "olcu", "dimensions", "length", "width", "height",
            "uzunluk", "genislik", "yukseklik",
        ],
        "safety": [
            "guvenlik", "safety", "uyari", "warning", "risk",
            "emergency stop", "e-stop", "stop",
        ],
        "maintenance": [
            "bakim", "maintenance", "servis", "service", "prosedur",
            "procedure",
        ],
        "communication": [
            "haberlesme", "communication", "ethernet", "usb", "ros",
            "rosserial", "mcu",
        ],
        "hmi": [
            "hmi", "panel", "button", "buton", "led", "light",
            "gosterge", "indicator",
        ],
        "water": ["su", "water", "ip", "ip54", "bilge", "pump", "pompa"],
        "comparison": [
            "karsilastir", "karsilastirma", "fark", "farki", "benzer",
            "ortak", "hangisi", "kiyasla", "compare", "comparison",
            "versus", "vs", "difference", "similarity",
        ],
    }

    QUERY_INTENT_EXPANSIONS = {
        "speed": "maximum speed top speed max speed velocity hiz maksimum hiz en yuksek hiz m/s km/h specifications",
        "payload": "payload maximum payload load capacity yuk tasima kapasitesi kg specifications",
        "weight": "weight mass agirlik kutle kg specifications",
        "runtime": "runtime run time operating time calisma suresi battery life hours specifications",
        "battery": "battery charge charger voltage current batarya sarj voltaj specifications",
        "dimensions": "dimensions length width height boyut olcu uzunluk genislik yukseklik specifications",
        "safety": "safety warning risk emergency stop guvenlik uyari",
        "maintenance": "maintenance service procedure bakim prosedur",
        "communication": "communication ethernet usb ros rosserial mcu haberlesme",
        "hmi": "HMI panel buttons LED indicators buton led gosterge",
        "water": "water IP rating bilge pump su gecisi pompa",
        "comparison": "compare comparison difference similarity fark karsilastirma ortak ozellik tablo",
    }

    
    def __init__(self):
        """RAG sistemini baÅŸlat"""
        print("ğŸ”„ RAG sistemi baÅŸlatÄ±lÄ±yor...")
        
        # Embedding modeli
        print("  - Embedding modeli yÃ¼kleniyor...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        
        # MongoDB baÄŸlantÄ±sÄ± (metadata, images)
        print("  - MongoDB baÄŸlanÄ±lÄ±yor...")
        self.db = MongoDBManager()
        
        # Qdrant baÄŸlantÄ±sÄ± (vector search)
        print("  - Qdrant baÄŸlanÄ±lÄ±yor...")
        self.vector_db = QdrantManager()
        
        # Llama LLM (Ollama Ã¼zerinden)
        print("  - Llama LLM baÄŸlanÄ±lÄ±yor...")
        try:
            self.llm = OllamaLLM(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.2,
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.0,
                num_predict=450,
            )
            print("âœ… RAG sistemi hazÄ±r")
        except Exception as e:
            print(f"âš ï¸ Ollama baÄŸlantÄ± hatasÄ±: {e}")
            print("  Ollama'nÄ±n Ã§alÄ±ÅŸtÄ±ÄŸÄ±ndan emin olun: ollama serve")
            print("  Model indirin: ollama pull llama3.1:8b")
            self.llm = None
        
        # Prompt ÅŸablonu
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=TURKISH_QA_PROMPT
        )

    def _fold_text(self, text: str) -> str:
        """Normalize Turkish/English query text for alias matching."""
        if not text:
            return ""

        folded = text.casefold()
        translation = str.maketrans({
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
            "â": "a",
            "î": "i",
            "û": "u",
        })
        folded = folded.translate(translation)
        return re.sub(r"\s+", " ", folded).strip()

    def _contains_alias(self, folded_query: str, alias: str) -> bool:
        folded_alias = self._fold_text(alias)
        if not folded_alias:
            return False

        pattern = rf"(?<![a-z0-9]){re.escape(folded_alias)}(?![a-z0-9])"
        return bool(re.search(pattern, folded_query))

    def _infer_query_categories(self, query: str) -> List[str]:
        """Infer all precise categories mentioned in the question."""
        folded_query = self._fold_text(query)
        if not folded_query:
            return []

        # More specific aliases must win before broad names such as "husky".
        matches = []
        for category, aliases in self.CATEGORY_ALIASES.items():
            if category not in CATEGORIES:
                continue
            for alias in aliases:
                if self._contains_alias(folded_query, alias):
                    matches.append((len(self._fold_text(alias)), category))

        if not matches:
            return []

        matches.sort(reverse=True)
        categories = []
        for _, category in matches:
            if category not in categories:
                categories.append(category)

        return categories

    def _infer_query_category(self, query: str) -> str:
        """Infer the strongest category from product/model names in the question."""
        categories = self._infer_query_categories(query)
        return categories[0] if categories else ""

    def _detect_query_intents(self, query: str) -> List[str]:
        folded_query = self._fold_text(query)
        if not folded_query:
            return []

        intents = []
        for intent, aliases in self.QUERY_INTENT_ALIASES.items():
            if any(self._contains_alias(folded_query, alias) for alias in aliases):
                intents.append(intent)

        return intents

    def _build_retrieval_queries(self, query: str, category: str = None) -> List[str]:
        """Create stable query variants so equivalent wording retrieves the same chunks."""
        variants = []

        def add_variant(value: str):
            value = " ".join(str(value or "").split())
            if value and value not in variants:
                variants.append(value)

        add_variant(query)

        inferred_category = category or self._infer_query_category(query)
        product_name = ""
        if inferred_category:
            product_name = CATEGORIES.get(inferred_category, {}).get("name", inferred_category)
            add_variant(f"{product_name} {query}")

        intents = self._detect_query_intents(query)
        for intent in intents:
            expansion = self.QUERY_INTENT_EXPANSIONS.get(intent, "")
            add_variant(f"{query} {expansion}")
            if product_name:
                add_variant(f"{product_name} {expansion}")

        return variants[:5]

    def _get_search_categories(self, query: str, category: str = None) -> List[str]:
        """Return category filters to search; multiple products enable multi-hop answers."""
        if category:
            return [category]

        inferred_categories = self._infer_query_categories(query)
        return inferred_categories or [None]

    def _primary_search_categories(self, search_categories: List[str]) -> List[str]:
        return search_categories or [None]

    def _prioritize_multi_category_vectors(
        self,
        vectors: List[Tuple[Dict, float]],
        categories: List[str],
        k: int,
    ) -> List[Tuple[Dict, float]]:
        """Keep evidence from each mentioned product before filling remaining slots."""
        concrete_categories = [cat for cat in categories if cat]
        if len(concrete_categories) < 2:
            return vectors

        selected = []
        selected_keys = set()

        for category in concrete_categories:
            for payload, score in vectors:
                if payload.get("category") != category:
                    continue
                key = (
                    (payload.get("source") or "").rstrip("/"),
                    payload.get("start_index", 0),
                    payload.get("mongodb_id", ""),
                    (payload.get("chunk_content") or "")[:500].strip(),
                )
                if key in selected_keys:
                    continue
                selected.append((payload, score))
                selected_keys.add(key)
                break

        for payload, score in vectors:
            if len(selected) >= max(k, len(concrete_categories)):
                break
            key = (
                (payload.get("source") or "").rstrip("/"),
                payload.get("start_index", 0),
                payload.get("mongodb_id", ""),
                (payload.get("chunk_content") or "")[:500].strip(),
            )
            if key in selected_keys:
                continue
            selected.append((payload, score))
            selected_keys.add(key)

        return selected

    def _is_insufficient_answer(self, answer: str) -> bool:
        """Model baglamda cevap bulamadigini soyluyorsa kaynak gostermeyi engelle."""
        if not answer:
            return True

        normalized = " ".join(answer.casefold().split())
        ascii_normalized = (
            normalized
            .replace("ç", "c")
            .replace("ğ", "g")
            .replace("ı", "i")
            .replace("ö", "o")
            .replace("ş", "s")
            .replace("ü", "u")
        )
        insufficient_markers = [
            "mevcut dokumanlarda bulamadim",
            "mevcut dokümanlarda bulamadım",
            "mevcut kaynaklarda bulamadim",
            "mevcut kaynaklarda bulamadım",
            "kaynaklarda bulamadim",
            "kaynaklarda bulamadım",
            "baglamda yok",
            "bağlamda yok",
            "baglamda bulunmuyor",
            "bağlamda bulunmuyor",
            "baglamda yer almiyor",
            "bağlamda yer almıyor",
            "baglamda gecmiyor",
            "bağlamda geçmiyor",
            "bilgi bulamadim",
            "bilgi bulamadım",
            "bilgi bulunmuyor",
            "bilgi bulunmamaktadir",
            "bilgi bulunmamaktadır",
            "cevabini bulamadim",
            "cevabını bulamadım",
            "cevabi bulunmuyor",
            "cevabı bulunmuyor",
            "dokumanlarda yer almiyor",
            "dokümanlarda yer almıyor",
            "dokumanlarda bulunmuyor",
            "dokümanlarda bulunmuyor",
            "dokumanlarda bulunmamaktadir",
            "dokümanlarda bulunmamaktadır",
            "dokumanlarda gecmiyor",
            "dokümanlarda geçmiyor",
            "net bir bilgi yok",
            "dogrudan bilgi yok",
            "doğrudan bilgi yok",
            "yeterli bilgi yok",
            "yeterli veri yok",
            "emin olmadigim",
            "emin olmadığım",
        ]
        return any(
            marker in normalized or marker in ascii_normalized
            for marker in insufficient_markers
        )

    def _filter_display_sources(self, sources: List[Dict], answer: str) -> List[Dict]:
        """UI'da yalnizca guvenilir ve cevabi destekleyen kaynaklari goster."""
        if self._is_insufficient_answer(answer):
            return []

        return [
            source for source in sources
            if float(source.get("similarity", 0.0)) >= SOURCE_DISPLAY_THRESHOLD
        ]
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        category: str = None,
        k: int = RETRIEVAL_K
    ) -> List[Tuple[Dict, float]]:
        """
        Sorguya en uygun chunk'ları getir (Qdrant'tan direkt chunk content kullanarak)
        
        Args:
            query: Kullanıcı sorusu
            category: Opsiyonel kategori filtresi
            k: Getirilecek chunk sayısı
            
        Returns:
            (chunk_dict, skor) tuple'larÄ±nÄ±n listesi
        """
        search_categories = self._get_search_categories(query, category)
        concrete_categories = [cat for cat in search_categories if cat]
        target_doc_count = max(k, len(concrete_categories))
        if concrete_categories and not category:
            print(f"  ℹ️ Sorgudan kategori tahmin edildi: {', '.join(concrete_categories)}")

        merged_vectors = {}
        variant_count = 0
        primary_categories = self._primary_search_categories(search_categories)
        for search_category in primary_categories:
            retrieval_queries = self._build_retrieval_queries(query, search_category)
            variant_count += len(retrieval_queries)
            for retrieval_query in retrieval_queries:
                # Sorgu iÃ§in embedding oluÅŸtur
                query_embedding = self.embeddings.embed_query(retrieval_query)

                # Qdrant'tan benzer vektÃ¶rleri ara
                similar_vectors = self.vector_db.search_similar(
                    query_vector=query_embedding,
                    category=search_category,
                    limit=max(k * 4, target_doc_count),
                    score_threshold=SIMILARITY_THRESHOLD
                )

                for payload, score in similar_vectors:
                    chunk_content = payload.get("chunk_content", "")
                    dedupe_key = (
                        (payload.get("source") or "").rstrip("/"),
                        payload.get("start_index", 0),
                        payload.get("mongodb_id", ""),
                        chunk_content[:500].strip(),
                    )
                    adjusted_score = float(score)
                    previous = merged_vectors.get(dedupe_key)
                    if not previous or adjusted_score > previous[1]:
                        merged_vectors[dedupe_key] = (payload, adjusted_score)

        if variant_count > 1:
            print(f"  ℹ️ {variant_count} sorgu varyantÄ± ile arama yapÄ±lÄ±yor")

        similar_vectors = sorted(
            merged_vectors.values(),
            key=lambda item: item[1],
            reverse=True,
        )
        similar_vectors = self._prioritize_multi_category_vectors(
            similar_vectors,
            primary_categories,
            target_doc_count,
        )
        
        # Qdrant payload'Ä±ndan chunk content'i direkt al (MongoDB'ye gitme)
        scored_docs = []
        seen_keys = set()
        for payload, score in similar_vectors:
            # â­ Qdrant'tan chunk content'i al
            chunk_content = payload.get("chunk_content")
            
            # EÄŸer chunk_content yoksa (eski veri), MongoDB'den al
            if not chunk_content:
                mongodb_id = payload.get("mongodb_id")
                if mongodb_id:
                    doc = self.db.collection.find_one({"_id": ObjectId(mongodb_id)})
                    if doc:
                        chunk_content = doc.get("content", "")
                        payload.setdefault("section_title", doc.get("metadata", {}).get("section_title", ""))
            
            images = payload.get("images", [])
            if not images:
                mongodb_id = payload.get("mongodb_id")
                if mongodb_id:
                    try:
                        doc = self.db.collection.find_one({"_id": ObjectId(mongodb_id)})
                        if doc:
                            images = doc.get("images", [])
                    except Exception:
                        images = []

            if chunk_content:
                normalized_source = (payload.get("source") or "").rstrip("/")
                start_index = payload.get("start_index", 0)
                mongodb_id = payload.get("mongodb_id")
                content_fingerprint = chunk_content[:500].strip()
                dedupe_key = (normalized_source, start_index, content_fingerprint)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                # Chunk'Ä± mock document biÃ§iminde oluÅŸtur
                chunk_dict = {
                    "content": chunk_content,
                    "source": payload.get("source", "Bilinmiyor"),
                    "category": payload.get("category", "Bilinmiyor"),
                    "section_title": payload.get("section_title", ""),
                    "start_index": start_index,
                    "mongodb_id": mongodb_id,
                    "image_hashes": payload.get("image_hashes", []),
                    "images": images
                }
                scored_docs.append((chunk_dict, score))
                if len(scored_docs) >= target_doc_count:
                    break

        if not scored_docs:
            print("âš ï¸ Ä°lgili dokÃ¼man bulunamadÄ±!")
            return []

        scored_docs = sorted(scored_docs, key=lambda item: item[1], reverse=True)
        scored_docs = self._prioritize_multi_category_vectors(
            scored_docs,
            search_categories,
            target_doc_count,
        )
        return scored_docs[:target_doc_count]
    
    def generate_answer(self, query: str, category: str = None) -> Dict:
        """
        Soru iÃ§in RAG pipeline'Ä± Ã§alÄ±ÅŸtÄ±r ve cevap Ã¼ret
        
        Args:
            query: KullanÄ±cÄ± sorusu
            category: Opsiyonel kategori filtresi
            
        Returns:
            Cevap bilgileri iÃ§eren dictionary
        """
        if not self.llm:
            return {
                "answer": "LLM servisi aktif deÄŸil. LÃ¼tfen Ollama'yÄ± baÅŸlatÄ±n.",
                "sources": [],
                "error": True
            }
        
        # 1. Ä°lgili dokÃ¼manlarÄ± getir
        print(f"\nğŸ” Soru: {query}")
        print("  DokÃ¼manlar aranÄ±yor...")
        
        relevant_docs = self.retrieve_relevant_documents(query, category)
        
        if not relevant_docs:
            return {
                "answer": "Bu sorunun cevabini mevcut dokumanlarda bulamadim. Lutfen sorunuzu yeniden formule etmeyi deneyin.",
                "sources": [],
                "error": False
            }
        
        print(f"  âœ… {len(relevant_docs)} ilgili dokÃ¼man bulundu")
        
        # 2. BaÄŸlam oluÅŸtur
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(relevant_docs, 1):
            safe_content = sanitize_retrieved_text(doc.get("content", ""))
            context_header = (
                f"Dokuman {i} | Kategori: {doc.get('category', 'Bilinmiyor')} "
                f"| Baslik: {doc.get('section_title', '') or 'Bilinmiyor'} "
                f"| Benzerlik: {score:.2f}"
            )
            context_parts.append(f"{context_header}\n{safe_content}")
            sources.append({
                "source": doc.get("source", "Bilinmiyor"),
                "category": doc.get("category", "Bilinmiyor"),
                "section_title": doc.get("section_title", ""),
                "similarity": float(score),
                "content_preview": safe_content[:200] + "...",
                "images": doc.get("images", [])  # GÃ¶rselleri ekle
            })
        
        context = build_untrusted_context_block(
            context_parts,
            max_chars=MAX_CONTEXT_CHARS,
            label="Technical Context",
        )
        
        # 3. LLM ile cevap Ã¼ret
        print("  ğŸ¤– Cevap oluÅŸturuluyor...")
        
        try:
            # Prompt'u formatla
            formatted_prompt = self.prompt.format(
                context=context,
                question=query
            )
            formatted_prompt = with_no_think(formatted_prompt)
            
            # LLM'den cevap al
            answer = self.llm.invoke(formatted_prompt)
            answer = strip_thinking(answer).strip()
            display_sources = self._filter_display_sources(sources, answer)
            
            return {
                "answer": answer,
                "sources": display_sources,
                "error": False
            }
        
        except Exception as e:
            print(f"  âŒ LLM hatasÄ±: {e}")
            return {
                "answer": f"Cevap Ã¼retilirken bir hata oluÅŸtu: {str(e)}",
                "sources": sources,
                "error": True
            }
    
    def close(self):
        """BaÄŸlantÄ±larÄ± kapat"""
        self.db.close()


# Test fonksiyonu
if __name__ == "__main__":
    # RAG sistemini test et
    rag = RAGSystem()
    
    # Test sorularÄ±
    test_questions = [
        "Warthog robotunun bakÄ±m prosedÃ¼rleri nelerdir?",
        "Warthog bataryasÄ± nasÄ±l ÅŸarj edilir?",
        "Warthog'un gÃ¼venlik Ã¶zellikleri neler?",
    ]
    
    print("\n" + "="*60)
    print("ğŸ§ª RAG SÄ°STEMÄ° TEST")
    print("="*60)
    
    for question in test_questions:
        result = rag.generate_answer(question)
        
        print(f"\n{'='*60}")
        print(f"â“ SORU: {question}")
        print(f"{'='*60}")
        print(f"\nğŸ’¡ CEVAP:\n{result['answer']}")
        
        if result['sources']:
            print(f"\nğŸ“š KAYNAKLAR ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  {i}. {source['source']}")
                print(f"     Kategori: {source['category']}")
                print(f"     Benzerlik: {source['similarity']:.2%}")
        
        print("\n" + "="*60)
    
    rag.close()
