# src/main.py
import os, json; from datetime import datetime
from .document_processor import DocumentProcessor
from .persona_analyzer import PersonaJobAnalyzer
from .query_generator import QueryGenerator
from .relevance_scorer import HybridRelevanceScorer
from .subsection_analyzer import SubSectionAnalyzer

def main(input_dir, output_dir, query_model_path, semantic_model_path):
    # 1. Initialize all modules
    doc_processor, persona_analyzer = DocumentProcessor(), PersonaJobAnalyzer()
    query_generator = QueryGenerator(query_model_path)
    relevance_scorer = HybridRelevanceScorer(semantic_model_path)
    subsection_analyzer = SubSectionAnalyzer()

    # 2. Load Input
    with open(os.path.join(input_dir, 'input.json'), 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 3. Run pipeline
    pdf_paths = [os.path.join(input_dir, doc['filename']) for doc in input_data['documents']]
    processed_docs = doc_processor.process_documents(pdf_paths)
    requirements = persona_analyzer.analyze(input_data['persona']['role'], input_data['job_to_be_done']['task'])
    ai_queries = query_generator.generate(input_data['persona']['role'], input_data['job_to_be_done']['task'])
    top_sections = relevance_scorer.score(processed_docs, requirements, ai_queries)
    analysis_results = subsection_analyzer.analyze(top_sections[:5], requirements, processed_docs)

    # 4. Format and write output
    final_output = {
        "metadata": {"input_documents": [doc['filename'] for doc in input_data['documents']], "persona": input_data['persona']['role'], "job_to_be_done": input_data['job_to_be_done']['task'], "processing_timestamp": datetime.now().isoformat()},
        "extracted_sections": [{'document': s['document'], 'section_title': s['title'], 'importance_rank': s['importance_rank'], 'page_number': s['page_number']} for s in analysis_results],
        "subsection_analysis": [{'document': s['document'], 'refined_text': s['refined_text'], 'page_number': s['page_number']} for s in analysis_results]
    }
    with open(os.path.join(output_dir, 'output.json'), 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    print(f"✅ Success! Output written to {os.path.join(output_dir, 'output.json')}")

if __name__ == '__main__':
    print("--- RUNNING IN LOCAL TEST MODE (FINAL HYBRID PIPELINE) ---")
    LOCAL_INPUT_DIR = 'input'
    LOCAL_OUTPUT_DIR = 'output'
    LOCAL_QUERY_MODEL_PATH = 'models/google-t5_t5-small'
    LOCAL_SEMANTIC_MODEL_PATH = f'models/{("BAAI/bge-base-en-v1.5").replace("/", "_")}'
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(LOCAL_QUERY_MODEL_PATH) or not os.path.exists(LOCAL_SEMANTIC_MODEL_PATH):
        print("❌ FATAL ERROR: A required model was not found. Run 'python src/download_models.py' first.")
    else:
        main(LOCAL_INPUT_DIR, LOCAL_OUTPUT_DIR, LOCAL_QUERY_MODEL_PATH, LOCAL_SEMANTIC_MODEL_PATH)