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
PAUSE = True

model = OllamaLLM(model="deepseek-r1:14b",
    options={"num_ctx": 8192},     #number of tokens an LLM accepts as input. Both system message and user message              
    base_url=os.environ["OLLAMA_HOST"],
    temperature=0.0,)

# robust path resolution
BASE_DIR = Path(__file__).resolve().parents[1]

file_path = BASE_DIR / "structured" 

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
    
if(not PAUSE):
    content = load("simplified_text.txt")
    response = requests.post("http://192.168.0.46:8001/spacy/split", json={"text":content})
    response_content = response.json()
    result = save(response_content, "spacy_split.json")
    print("✅✅✅✅✅✅✅✅SPACY RESPONSE ===",result)

if(not PAUSE):
    response_content = json.loads(load("spacy_split.json"))
    # Initialize tracking list
    sentences = [s.strip() for s in response_content["sentences"] if s.strip()]
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
    
        # Call simplification function
        raw_new_sentences_text = simplify_text_stage2(sentence, model)
        new_sentences_text = remove_think_block(raw_new_sentences_text)
        print("🧾",new_sentences_text)

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

    save(sentences,"simple_sentences.txt")

if(not PAUSE):
    sentences = json.loads(load("simple_sentences.txt"))
    ontologies = None
    # Final cleaned list
    print("🧾🧾🧾🧾🧾🧾🧾🧾 Final simplified sentence list:", sentences)

    all_atomic_jsons = []                # <--- collect here

    for s in sentences:
        print("-🧾", s)

        raw_kg = create_knowledge_ontology(s, model)
        kg_txt = remove_think_block(raw_kg).strip()

        # the LLM sometimes wraps the JSON in ```json … ``` fences → strip them
        if kg_txt.startswith("```"):
            kg_txt = kg_txt.replace("```json", "").replace("```", "").strip()

        print("-😁", kg_txt)

        try:
            kg_obj = json.loads(kg_txt)  # turn the string into a Python dict
            all_atomic_jsons.append(kg_obj)
        except json.JSONDecodeError as e:
            print("⚠️  could not parse KG JSON:", e)
        #    optional: re-prompt the model here

    save(all_atomic_jsons,"atomic_sentences.txt")


    
all_atomic_jsons = json.load(load("atomic_sentences.txt"))
print("---------------------------------------------------------- ATOMIC SENTENCES:",all_atomic_jsons)
file_content = load("output.json")
# once the loop is done, fuse them
integrated_graph = fuse_atomic_graphs(file_content, all_atomic_jsons, model)
print(json.dumps(integrated_graph, indent=2, ensure_ascii=False))
