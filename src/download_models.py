# src/download_models.py
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from spacy.cli import download
import nltk

SEMANTIC_MODEL = 'BAAI/bge-base-en-v1.5'
QUERY_MODEL = 'google-t5/t5-small'
SPACY_MODEL = 'en_core_web_sm'

def main():
    print("--- Downloading all required models for offline use ---")
    os.makedirs('models', exist_ok=True)
    
    bge_path = f'models/{SEMANTIC_MODEL.replace("/", "_")}'
    if not os.path.exists(bge_path):
        print(f"\nDownloading semantic model '{SEMANTIC_MODEL}'...")
        SentenceTransformer(SEMANTIC_MODEL).save(bge_path)
        print(f"✅ BGE Model saved to: {bge_path}")

    t5_path = f'models/{QUERY_MODEL.replace("/", "_")}'
    if not os.path.exists(t5_path):
        print(f"\nDownloading query model '{QUERY_MODEL}'...")
        T5Tokenizer.from_pretrained(QUERY_MODEL).save_pretrained(t5_path)
        T5ForConditionalGeneration.from_pretrained(QUERY_MODEL).save_pretrained(t5_path)
        print(f"✅ T5 Model saved to: {t5_path}")

    print(f"\nChecking/Downloading spaCy model '{SPACY_MODEL}'...")
    try: download(SPACY_MODEL)
    except SystemExit: print(f"✅ spaCy model '{SPACY_MODEL}' likely already exists.")
    except Exception as e: print(f"❌ spaCy model download failed: {e}")
    
    print("\nDownloading NLTK data...")
    nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)
    print("✅ All downloads complete.")

if __name__ == "__main__":
    main()