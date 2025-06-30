import getpass
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import spacy
import json
import requests
from LLMs import simplify_text, remove_think_block, simplify_text_stage2, create_knowledge_ontology, fuse_atomic_graphs

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


if os.environ["LANGSMITH_TRACING"] == "true":
    if "LANGSMITH_API_KEY" not in os.environ:
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
            prompt="Enter your LangSmith API key (optional): "
        )
    if "LANGSMITH_PROJECT" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
        if not os.environ.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "default"

os.environ["OLLAMA_HOST"] = os.environ["OLLAMA_HOST_PC"]

model = OllamaLLM(model="deepseek-r1:14b",
    options={"num_ctx": 8192},     #number of tokens an LLM accepts as input. Both system message and user message              
    base_url=os.environ["OLLAMA_HOST"],
    temperature=0.0,)

# robust path resolution
BASE_DIR = Path(__file__).resolve().parents[1]

file_path = BASE_DIR / "structured" / "output.json"

with file_path.open(encoding="utf-8") as f:
    file_content = f.read()

print("✅✅✅✅✅✅✅✅ Input: \n",file_content)
raw_output = simplify_text(file_content, model)
clean_output = remove_think_block(raw_output)
print("✅✅✅✅✅✅✅✅ Output: \n",clean_output)
    
response = requests.post("http://192.168.0.46:8001/spacy/split", json={"text":clean_output})
response_content = response.json()
print("✅✅✅✅✅✅✅✅SPACY RESPONSE ===",response_content)

# Initialize tracking list
sentences = [s.strip() for s in response_content["sentences"] if s.strip()]
i = 0

while i < len(sentences):
    sentence = sentences[i]
    
    # Call simplification function
    raw_new_sentences_text = simplify_text_stage2(sentence, model)
    new_sentences_text = remove_think_block(raw_new_sentences_text)
    # print("🧾",new_sentences_text)

    # Split the result by newlines or sentence delimiters
    new_sentences = [s.strip() for s in new_sentences_text.split('\n') if s.strip()]

    if len(new_sentences) == 1 and new_sentences[0] == sentence:
        i += 1
        continue

    # If the result has more than one sentence, replace
    if len(new_sentences) > 1:
        # Remove the original sentence
        sentences.pop(i)
        # Insert the new ones at the same index
        for new_sentence in reversed(new_sentences):
            sentences.insert(i, new_sentence)
        
        i += len(new_sentences) 
    else:
        i += 1  # move on to the next one
ontologies = None
# Final cleaned list
print("🧾🧾🧾🧾🧾🧾🧾🧾 Final simplified sentence list:")
for s in sentences:
    print("-🧾", s)
    raw_kg = create_knowledge_ontology(s, model)
    kg = remove_think_block(raw_kg)
    print("-😁", kg)

integrated_graph = fuse_atomic_graphs(file_content, all_atomic_jsons, model)
print(json.dumps(integrated_graph, indent=2, ensure_ascii=False))