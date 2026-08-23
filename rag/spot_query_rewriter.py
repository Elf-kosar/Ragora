"""Language-bridging query rewriting for Spot manuals.

Spot's source manuals are English, while the shared Ragora UI accepts Turkish
questions. This module keeps retrieval-time language handling separate from the
Spot pipeline adapter so the adapter remains a thin facade over the native Spot
project.
"""
from __future__ import annotations

import unicodedata
from typing import List

from langchain_core.prompts import PromptTemplate

from rag.model_utils import strip_thinking, with_no_think


SPOT_SEARCH_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Rewrite the user question as one concise English technical search query for Boston Dynamics Spot manuals.

Rules:
- Preserve exact product names such as Spot, Spot Arm, Spot CAM, Dock, battery, power supply, E-Stop.
- Preserve numbers, units, error codes and part names.
- Do not answer the question.
- Return only the rewritten search query.

User question: {question}

English search query:""",
)


class SpotQueryRewriter:
    """Builds retrieval queries for English Spot manuals from multilingual input."""

    def __init__(self, llm=None):
        self.llm = llm
        self._cache = {}

    def build_retrieval_queries(self, query: str) -> List[str]:
        candidates = [query]
        rewritten = self.rewrite_for_search(query)
        if rewritten:
            candidates.append(rewritten)

        normalized = self.normalize_text(" ".join(candidates))
        candidates.extend(self.domain_expansions(normalized, query, rewritten or query))

        deduped = []
        seen = set()
        for item in candidates:
            clean = " ".join(str(item or "").split())
            key = self.normalize_text(clean)
            if clean and key not in seen:
                seen.add(key)
                deduped.append(clean)
        return deduped

    def rewrite_for_search(self, query: str) -> str:
        cache_key = self.normalize_text(query)
        if cache_key in self._cache:
            return self._cache[cache_key]

        rewritten = ""
        if self.llm and hasattr(self.llm, "invoke") and self.looks_non_english(query):
            try:
                prompt = SPOT_SEARCH_QUERY_PROMPT.format(question=query)
                rewritten = strip_thinking(self.llm.invoke(with_no_think(prompt))).strip()
                rewritten = self.clean_rewritten_query(rewritten)
            except Exception:
                rewritten = ""

        if not rewritten:
            rewritten = self.lexical_english_rewrite(query)

        if self.normalize_text(rewritten) == cache_key:
            rewritten = ""
        self._cache[cache_key] = rewritten
        return rewritten

    def clean_rewritten_query(self, text: str) -> str:
        text = (text or "").strip().strip('"').strip("'")
        for prefix in ("English search query:", "Search query:", "Query:"):
            if text.casefold().startswith(prefix.casefold()):
                text = text[len(prefix):].strip()
        return text.splitlines()[0].strip()[:240]

    def lexical_english_rewrite(self, query: str) -> str:
        normalized = self.normalize_text(query)
        product = "Spot Arm" if self.query_mentions_arm(query) else "Spot robot"
        concepts = []
        concept_map = (
            (("maksimum", "maximum", "hiz", "speed", "velocity"), "maximum speed velocity"),
            (("agirlik", "weight", "mass"), "weight mass specifications"),
            (("batarya", "pil", "battery", "runtime"), "battery runtime charge"),
            (("sarj", "charger", "guc kaynagi", "power supply"), "charger power supply"),
            (("guvenlik", "safety", "e-stop", "estop", "stop"), "safety emergency stop warning"),
            (("sicaklik", "temperature", "isi", "isinma"), "operating temperature storage temperature"),
            (("boyut", "olcu", "ebat", "dimension", "size"), "dimensions size length width height"),
            (("egim", "merdiven", "engel", "slope", "stairs", "obstacle"), "slope stairs obstacle terrain mobility"),
            (("kamera", "camera", "cam"), "camera Spot CAM"),
            (("dock", "istasyon", "station"), "dock station"),
        )
        for triggers, expansion in concept_map:
            if any(term in normalized for term in triggers):
                concepts.append(expansion)
        return " ".join([product, *concepts]).strip()

    def domain_expansions(self, normalized: str, original_query: str, search_query: str) -> List[str]:
        subject = "Spot Arm" if self.query_mentions_arm(original_query) else "Spot robot"
        expansions = []
        if any(term in normalized for term in ("hiz", "speed", "velocity", "maksimum", "maximum")):
            if self.query_mentions_arm(original_query):
                expansions.append(f"{subject} end-effector maximum speed velocity limitation 0.75 m/s")
            else:
                expansions.extend(
                    [
                        f"{subject} walk gait maximum speed velocity limitation 1.6 m/s",
                        f"{search_query} speed setting maximum velocity Fast Med Slow",
                    ]
                )
        if any(term in normalized for term in ("agirlik", "weight", "mass")):
            expansions.append(f"{subject} weight mass total weight kg specifications")
        if any(term in normalized for term in ("batarya", "pil", "battery", "charge", "sarj")):
            expansions.append(f"{subject} battery charge charger runtime voltage current")
        if any(term in normalized for term in ("guvenlik", "safety", "e-stop", "estop", "stop")):
            expansions.append(f"{subject} safety warning emergency stop e-stop stopping distance")
        return expansions

    def preferred_doc_prefixes(self, query: str) -> List[str]:
        normalized = self.normalize_text(query)
        if self.query_mentions_arm(query):
            return ["spot-arm-user-manual-en", "spot-user-manual-en"]
        if any(term in normalized for term in ("kamera", "camera", "cam")):
            return ["spot-cam-2-user-manual-en", "spot-user-manual-en"]
        if any(term in normalized for term in ("dock", "istasyon", "station")):
            return ["spot-dock-user-manual-en", "spot-station-user-manual-en", "spot-user-manual-en"]
        if any(term in normalized for term in ("power supply", "guc kaynagi", "charger", "sarj")):
            return ["spot-power-supply-user-manual-en", "spot-dock-user-manual-en", "spot-user-manual-en"]
        return ["spot-user-manual-en", "spot-arm-user-manual-en"]

    def query_mentions_arm(self, query: str) -> bool:
        normalized = self.normalize_text(query)
        return any(term in normalized for term in ("spot arm", " arm", "kol", "gripper", "end-effector", "manipulator"))

    def answer_keywords(self, query: str) -> List[str]:
        normalized = self.normalize_text(query)
        keywords = []
        if any(term in normalized for term in ("hiz", "speed", "velocity", "maksimum", "maximum")):
            keywords.extend(["maximum velocity", "maximum speed", "1.6 m/s", "0.75 m/s", "speed setting"])
        if any(term in normalized for term in ("agirlik", "weight", "mass")):
            keywords.extend(["total weight", "mass", "weight", "kg"])
        if any(term in normalized for term in ("batarya", "pil", "battery", "runtime")):
            keywords.extend(["battery", "runtime", "charge", "hours"])
        if any(term in normalized for term in ("guvenlik", "safety", "e-stop", "estop", "stop")):
            keywords.extend(["safety", "warning", "emergency stop", "e-stop", "stopping"])
        return keywords

    def answer_language(self, question: str) -> str:
        return "Turkish" if self.looks_non_english(question) else "English"

    def looks_non_english(self, text: str) -> bool:
        normalized = self.normalize_text(text)
        turkish_markers = (
            " nedir", " nasil", " ne kadar", " kac", " hangi", " mi", " mu",
            " hiz", " agirlik", " guvenlik", " sarj", " batarya", " pil", " kol",
        )
        return any(ch in (text or "") for ch in "\u0131\u0130\u011f\u011e\u00fc\u00dc\u015f\u015e\u00f6\u00d6\u00e7\u00C7") or any(
            marker in f" {normalized} " for marker in turkish_markers
        )

    def normalize_text(self, text: str) -> str:
        replacements = str.maketrans(
            {
                "\u0131": "i",
                "\u0130": "I",
                "\u011f": "g",
                "\u011e": "G",
                "\u00fc": "u",
                "\u00dc": "U",
                "\u015f": "s",
                "\u015e": "S",
                "\u00f6": "o",
                "\u00d6": "O",
                "\u00e7": "c",
                "\u00C7": "C",
            }
        )
        text = (text or "").translate(replacements)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return " ".join(text.casefold().split())
