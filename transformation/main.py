import getpass
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

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

model = ChatOllama(model="deepseek-r1:7b",
    options={"num_ctx": 8192},     #number of tokens an LLM accepts as input. Both system message and user message              
    base_url=os.environ["OLLAMA_HOST"],
    temperature=0.2,)

# robust path resolution
BASE_DIR = Path(__file__).resolve().parents[1]

file_path = BASE_DIR / "structured" / "output.json"

with file_path.open(encoding="utf-8") as f:
    file_content = f.read()

messages_template = ([
    ("system", "You are a helpful extraction assistant. You are going to be given a json "
    "file representing a preproccesed {file_type}."
    "I want you to go over whole json file, consider how would a knowledge graph look based on this text."
    "Extract the nodes as ontology triples, subject, predicate and object based on this text. "
    "Example: Document Title: General Terms of Vehicle Insurance"
    "Section 3.1 - Covered Scenarios"
    "Paragraph 1 - Collision with Another Vehicle"
    "If the insured Vehicle is involved in a collision with another road-legal Vehicle andthe Driver of the insured Vehicle "
    "holds a valid driver's License, the insurance covers the full cost of repairs to the insured Vehicle, minus the "
    "deductible of 200 Euros."
    "Expected Answer:"
    "[node]"
    "id = n1"
    "name = Vehicle Insurance"
    "[node]"
    "id = n2"
    "name = Collision with another vehicle"
    "[node]"
    "id = n3"
    "name = Natural Disasters"
    "[node]"
    "id = n4"
    "name = Floods"
    "[node]"
    "id = n5"
    "name = AND"
    "[node]"
    "id = n6"
    "name = Collides with another vehicle"
    "[node]"
    "id = n7"
    "name = driver has drivers licence"
    "[node]"
    "id = n8"
    "name = Insured Vehicle"
    "[node]"
    "id = n9"
    "name = Another Vehicle"
    "[edge]"
    "name = covers_scenario"
    "from = n1"
    "to = n2"
    "[edge]"
    "name = covers_scenario"
    "from = n1"
    "to = n3"
    "[edge]"
    "name = covers_scenario"
    "from = n3"
    "to = n4"
    "[edge]"
    "name = is_insured"
    "from = n2"
    "to = n5"
    "[edge]"
    "name = fulfills_condition"
    "from = n6"
    "to = n5"
    "[edge]"
    "name = fulfills_condition"
    "from = n7"
    "to = n5"
    "[edge]"
    "name = fulfills_condition"
    "from = n8"
    "to = n6"      
    "[edge]"
    "name = collides"
    "from = n8"
    "to = n9"),
    ("user", "{file}")
])
print(messages_template)

prompt_tmpl = ChatPromptTemplate.from_messages(messages_template)
print("✅ Built prompt-template")

messages = prompt_tmpl.format_messages(file_type=".pdf", file=file_content)

response = model.invoke(messages)
print(response)