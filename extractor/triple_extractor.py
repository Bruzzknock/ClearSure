import os
import re
import requests
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# Load once when script starts
extractor = pipeline("text2text-generation", model="Babelscape/rebel-large")

def extract_triples_with_llm(text):
    print("🧠 Extracting triples locally with rebel-large...")
    result = extractor(text, max_length=512, clean_up_tokenization_spaces=True)[0]["generated_text"]
    print("✅ Raw Output:", result)
    return result


def parse_triples(raw_output):
    triples = []
    lines = raw_output.strip().split("\n")
    for line in lines:
        match = re.match(r'\(?\s*"?([^",]+)"?\s*,\s*"?([^",]+)"?\s*,\s*"?([^",]+)"?\s*\)?', line)
        if match:
            triples.append((match.group(1), match.group(2), match.group(3)))
    return triples

from urllib.parse import quote

def parse_rebel_output(raw_output):
    """
    Parse REBEL output: subject  object  predicate → convert to (subject_URI, predicate_URI, object_Literal)
    """
    triples = []
    parts = raw_output.strip().split("  ")  # REBEL uses double-space separator
    for i in range(0, len(parts) - 2, 3):
        subj = quote(parts[i].strip())          # URI-safe subject
        obj = parts[i + 1].strip()              # Keep object as is (Literal)
        pred = quote(parts[i + 2].strip())      # URI-safe predicate
        triples.append((subj, pred, obj))
    return triples
