# src/document_processor.py
import re
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber

class DocumentProcessor:
    def process_documents(self, pdf_paths: List[str]) -> Dict[str, Any]:
        print(f"📚 Processing {len(pdf_paths)} documents...")
        processed_docs = {}
        for pdf_path in pdf_paths:
            try:
                doc_name = Path(pdf_path).name
                doc_data = {'file_name': doc_name, 'sections': []}
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        doc_data['sections'].extend(self._extract_sections_from_page(page.extract_text(x_tolerance=2) or "", page_num))
                if doc_data['sections']: processed_docs[doc_name] = doc_data
            except Exception as e: print(f"❌ Error processing {pdf_path}: {e}")
        return processed_docs

    def _extract_sections_from_page(self, text: str, page_num: int) -> List[Dict]:
        sections, lines = [], text.split('\n')
        current_title, current_content = f"Content from page {page_num}", []
        for line in lines:
            stripped = line.strip()
            if not stripped: continue
            if self._is_section_header(stripped):
                if current_content:
                    content_str = ' '.join(current_content).strip()
                    sections.append({'title': current_title, 'content': content_str, 'page_number': page_num, 'word_count': len(content_str.split())})
                current_title, current_content = stripped, []
            else:
                current_content.append(stripped)
        if current_content:
            content_str = ' '.join(current_content).strip()
            sections.append({'title': current_title, 'content': content_str, 'page_number': page_num, 'word_count': len(content_str.split())})
        return [s for s in sections if s['word_count'] > 5]

    def _is_section_header(self, line: str) -> bool:
        if len(line) < 3 or len(line) > 150: return False
        if re.match(r'^[0-9]+\.[0-9. ]*|^[IVX]+\. |^[A-Z]\. ', line): return True
        if line.isupper() and len(line.split()) < 10: return True
        if line.istitle() and len(line.split()) < 10 and not line.endswith(('.', ':', ',')): return True
        common_keywords = ['introduction', 'methodology', 'results', 'discussion', 'conclusion', 'abstract', 'background', 'overview', 'appendix', 'ingredients', 'instructions']
        if any(line.lower().startswith(keyword) for keyword in common_keywords): return True
        return False