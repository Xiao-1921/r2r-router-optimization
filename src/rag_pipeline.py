"""
RAG Retrieval Pipeline for R2R
==============================
Cascading query extraction, Wikipedia opensearch API,
and embedding-based similarity filtering.

Usage:
    from rag_pipeline import RAGRetriever, SimilarityFilter

    retriever = RAGRetriever()
    sim_filter = SimilarityFilter(embed_model)

    passages = retriever.retrieve(question)
    filtered, sims = sim_filter.filter_passages(question, passages)
"""

import re
import time
import requests
import numpy as np


class RAGRetriever:
    """Retrieves passages from Wikipedia using cascading query extraction."""

    def __init__(self, top_k=3):
        import wikipediaapi
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="R2R-NLP-CSCI544/1.0", language="en"
        )
        self.top_k = top_k
        self.stopwords = {
            "what","which","who","whom","where","when","why","how",
            "is","are","was","were","the","a","an","of","in","to",
            "for","and","or","that","this","with","from","by","on",
            "at","does","did","do","has","have","had","be","been",
            "being","it","its","not","no","can","could","would",
            "should","will","shall","may","might","about","most",
            "following","true","false","correct","incorrect","many",
            "much","some","any","all","each","every","both","few",
            "more","other","than","then","also","just","only",
            "very","well","even","still","already","often","never",
            "always","sometimes","usually","really","actually",
            "between","during","before","after","above","below",
            "under","over","into","through","against","among",
            "these","those","such","like","used","known","called",
        }

    def extract_queries(self, question):
        """Extract search queries using cascading strategy:
        1. Named entities (capitalized phrases)
        2. Content keywords (stopword-filtered)
        3. Cleaned question fallback
        """
        queries = []

        # Strategy 1: Named entities
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        entities = re.findall(cap_pattern, question[:500])
        words = question[:500].split()
        starters = set()
        for i, w in enumerate(words):
            if i == 0 or (i > 0 and words[i-1].endswith(('?', '.', '!'))):
                starters.add(w.strip("?,.'\""))
        entities = [e for e in entities if e.split()[0] not in starters]
        if entities:
            queries.append(" ".join(entities[:3]))
            for e in entities[:2]:
                if len(e) > 2:
                    queries.append(e)

        # Strategy 2: Content keywords
        all_words = re.findall(r'\b[A-Za-z]+\b', question[:500])
        keywords = [w for w in all_words if w.lower() not in self.stopwords and len(w) > 2]
        if keywords:
            queries.append(" ".join(keywords[:5]))

        # Strategy 3: Cleaned question
        clean_q = re.sub(
            r'^(what|which|who|where|when|how|why|is|are|was|were|does|did|do)\s+',
            '', question[:200].lower(), flags=re.IGNORECASE
        ).strip('?. ')
        if clean_q and len(clean_q) > 5:
            queries.append(clean_q[:80])

        # Deduplicate
        seen = set()
        unique = []
        for q in queries:
            ql = q.lower().strip()
            if ql not in seen and len(ql) > 2:
                seen.add(ql)
                unique.append(q)
        return unique[:4]

    def search_wikipedia(self, query):
        """Search Wikipedia using opensearch API."""
        results = []
        try:
            params = {
                "action": "opensearch",
                "search": query,
                "limit": 3,
                "namespace": 0,
                "format": "json",
            }
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params, timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                for title in (data[1] if len(data) > 1 else [])[:3]:
                    page = self.wiki.page(title)
                    if page.exists() and page.summary:
                        results.append((title, page.summary[:600]))
        except Exception:
            pass
        return results

    def retrieve(self, question):
        """Retrieve passages for a question.
        Returns list of (title, text) tuples.
        """
        queries = self.extract_queries(question)
        passages, seen = [], set()
        for query in queries:
            for title, text in self.search_wikipedia(query):
                if title.lower() not in seen:
                    seen.add(title.lower())
                    passages.append((title, text))
            if len(passages) >= self.top_k:
                break
            time.sleep(0.1)
        return passages[:self.top_k]


class SimilarityFilter:
    """Filters retrieved passages using embedding-based cosine similarity."""

    def __init__(self, embed_model, threshold=0.3):
        self.model = embed_model
        self.threshold = threshold

    def filter_passages(self, question, passages):
        """Filter passages by cosine similarity to the question.
        Returns (filtered_passages, similarities).
        """
        if not passages:
            return [], []
        q_emb = self.model.encode(question[:512], normalize_embeddings=True)
        filtered, sims = [], []
        for title, text in passages:
            p_emb = self.model.encode(text[:300], normalize_embeddings=True)
            sim = float(np.dot(q_emb, p_emb))
            sims.append(sim)
            if sim >= self.threshold:
                filtered.append((title, text))
        return filtered, sims


def get_rag_context(question, retriever, sim_filter, max_chars=1500):
    """Convenience function: retrieve and filter in one call.
    Returns context string (empty if nothing passes filter).
    """
    passages = retriever.retrieve(question)
    filtered, _ = sim_filter.filter_passages(question, passages)
    if filtered:
        parts = [f"[{title}]\n{text}" for title, text in filtered]
        context = "\n\n".join(parts)
        return context[:max_chars] if len(context) > max_chars else context
    return ""
