import getpass
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import spacy
import json
import requests
from LLMs import simplify_text, remove_think_block, extract_entities, create_knowledge_ontology, fuse_atomic_graphs
from run_pipeline import load_and_push, clear_database

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
PAUSE = True

model = OllamaLLM(model="deepseek-r1:14b",
    options={"num_ctx": 8192},     #number of tokens an LLM accepts as input. Both system message and user message              
    base_url=os.environ["OLLAMA_HOST"],
    temperature=0.0,)

# robust path resolution
BASE_DIR = Path(__file__).resolve().parents[1]

file_path = BASE_DIR / "structured" 
OUT_PATH  = BASE_DIR / "structured" / "import_kg.cypher"

def save(text: str, file: str) -> str:
    output_file = file_path / file
    if isinstance(text, (dict, list)):
        text_str = json.dumps(text, ensure_ascii=False, indent=2)
    else:
        text_str = str(text)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(text_str)
    return text_str

def load(file: str) -> str:
    input_file = file_path / file
    with input_file.open(encoding="utf-8") as f:
        file_content = f.read()
    return file_content

if(not PAUSE):
    content = load("output.json")
    print("✅✅✅✅✅✅✅✅ Input: \n",content)
    raw_output = simplify_text(content, model)
    clean_output = remove_think_block(raw_output)
    result = save(clean_output, "simplified_text.txt")
    print("✅✅✅✅✅✅✅✅ Output: \n",result)

if(PAUSE):
    simplified = load("simplified_text.txt")
    print("✅✅✅✅✅✅✅✅ Input: \n",simplified)
    raw_output = create_knowledge_ontology(simplified,model)
    clean_output = remove_think_block(raw_output)
    result = save(clean_output, "final_kg.json")
    print("✅✅✅✅✅✅✅✅ Inspected text: \n",result)
    
    
clear_database(drop_meta=True)           # wipe
load_and_push(save_to=OUT_PATH)          # reload + save copy
print("✅ Graph ingested and written to", OUT_PATH)