# src/query_generator.py
import re; from typing import List; from transformers import T5ForConditionalGeneration, T5Tokenizer
class QueryGenerator:
    def __init__(self, model_path: str):
        print(f"🧠 Loading Query Generator from {model_path}..."); self.tokenizer = T5Tokenizer.from_pretrained(model_path); self.model = T5ForConditionalGeneration.from_pretrained(model_path)
    def generate(self, persona: str, job: str) -> List[str]:
        prompt = f"generate search queries for: As a {persona}, I need to {job}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=128, num_beams=5, num_return_sequences=1, early_stopping=True)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        queries = [q.strip() for q in re.split(r',|;', text) if q.strip()]; queries.append(job)
        print(f"   🤖 Generated AI queries: {queries}"); return list(set(queries))