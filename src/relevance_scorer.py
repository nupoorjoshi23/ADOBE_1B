# src/relevance_scorer.py
from typing import Dict, Any, List; import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

class HybridRelevanceScorer:
    def __init__(self, ranking_model_path: str):
        print(f"🛰️ Loading Semantic Ranker from {ranking_model_path}..."); self.encoder = SentenceTransformer(ranking_model_path); self.tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
    def score(self, documents: Dict, requirements: Dict, ai_queries: List[str]) -> List[Dict]:
        print("📊 Calculating hybrid relevance scores..."); texts, metadata = [], []
        for doc_name, doc_data in documents.items():
            for section in doc_data['sections']: texts.append(section['content']); metadata.append({'document': doc_name, **section})
        if not texts: return []
        
        # Keyword scoring
        tfidf_matrix = self.tfidf.fit_transform(texts)
        query_tfidf = self.tfidf.transform([" ".join(requirements['all_keywords'])])
        keyword_scores = cosine_similarity(tfidf_matrix, query_tfidf).flatten()
        
        # Semantic scoring
        query_embeds = self.encoder.encode([f"Represent this sentence: {q}" for q in ai_queries], convert_to_tensor=True)
        section_embeds = self.encoder.encode(texts, convert_to_tensor=True)
        semantic_scores = np.max(util.cos_sim(query_embeds, section_embeds).cpu().numpy(), axis=0)

        scored_sections = []
        for i, meta in enumerate(metadata):
            score = (0.6 * semantic_scores[i]) + (0.4 * keyword_scores[i]) # Fused score
            if 50 <= meta['word_count'] <= 600: score *= 1.1 # Length bonus
            if any(kw in meta['title'].lower() for kw in requirements['weighted_keywords']): score *= 1.3 # Title bonus
            scored_sections.append({**meta, 'relevance_score': score})
        
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        for rank, section in enumerate(scored_sections, 1): section['importance_rank'] = rank
        return scored_sections
