# src/subsection_analyzer.py
import re
from typing import Dict, Any, List
from nltk.tokenize import sent_tokenize

class SubSectionAnalyzer:
    def analyze(self, top_sections: List[Dict], requirements: Dict, documents: Dict) -> List[Dict]:
        print("🔍 Analyzing sub-sections for final output...")
        results = []
        for section in top_sections:
            content = self._find_section_content(documents, section['document'], section['title'], section['page_number'])
            if content:
                results.append({**section, 'refined_text': self._extract_key_passages(content, requirements)})
        return results
    def _find_section_content(self, documents: Dict, doc_name: str, title: str, page: int) -> str:
        for section in documents.get(doc_name, {}).get('sections', []):
            if section['title'] == title and section['page_number'] == page: return section['content']
        return ""
    def _extract_key_passages(self, content: str, requirements: Dict) -> str:
        sentences = sent_tokenize(content)
        keywords = requirements.get('all_keywords', [])
        if not sentences or not keywords: return " ".join(content.split()[:150])
        scored = sorted([(s, sum(1 for kw in keywords if kw in s.lower())) for s in sentences], key=lambda x: x[1], reverse=True)
        return " ".join([s[0] for s in scored[:5]])