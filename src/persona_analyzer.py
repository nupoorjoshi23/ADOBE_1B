import re; from typing import Dict, Any, List; import spacy
from nltk.corpus import stopwords

class PersonaJobAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.domain_keywords = {
            'academic_research': ['methodology', 'literature', 'review', 'research', 'study', 'analysis', 'findings', 'results', 'discussion'],
            'business_analysis': ['revenue', 'profit', 'financial', 'market', 'strategy', 'competitive', 'growth', 'investment', 'performance'],
            'education': ['concept', 'theory', 'principle', 'explanation', 'example', 'study', 'learn'],
            'hr_forms': ['form', 'fillable', 'onboarding', 'compliance', 'signature', 'manage', 'create']
        }
    def analyze(self, persona: str, job: str) -> Dict[str, Any]:
        print("👤 Analyzing persona and job requirements...")
        job_lower = job.lower()
        doc = self.nlp(job)
        keywords = {chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) < 4}
        keywords.update({token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop})
        weighted_keywords = {kw: 2.0 if kw in {t.text.lower() for t in doc if t.pos_ == 'NOUN'} else 1.0 for kw in keywords}
        domain = next((d for d, k in self.domain_keywords.items() if any(w in job_lower for w in k)), 'general')
        return {'all_keywords': list(keywords), 'domain_focus': domain, 'weighted_keywords': weighted_keywords}