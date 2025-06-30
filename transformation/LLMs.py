from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import re
import json

def simplify_text(text: str, model) -> str:
    prompt_template = PromptTemplate.from_template(
"Task: Re-format the sentence below so it is easier for a Named-Entity-Recognition (NER) model to process."
"RULES:"
"Do not add, remove, or alter factual content."
"Do not paraphrase key terms (e.g. proper names, dates, numbers, durations, organisations)."
"Break up long or nested clauses into short, declarative sentences."
"Preserve original chronology and relationships."
"One fact = one sentence. No commas. No relatives (who, which, that). No parentheses. No clause-joining conjunctions."
"Avoid commas unless you are producing a true list; prefer multiple simple sentences instead of one complex sentence."
"Keep ALL information! Do not lose information. Keep ALL facts"
"Add a sentence to preserve that temporal relationship or keep them in order."
"Return only the reformatted sentences, one per line without any extra text."
"Example:"
"Input sentence: "
"Throughout the late afternoon of 14 October 2024—after the International Association for Climate Economics (IACE)"
" released its 312-page report titled “Carbon Costs and the Tomorrow Market”—Dr. María-José Fernández, who at 16 years old won"
" the 2008 Intel Science Award and now heads the quantum-finance lab at MIT, told 27 delegates from 9 EU states that if global"
" CO₂ emissions exceed 56.3 gigatonnes by 2030, the projected “social cost of carbon” could climb above $173.45 per metric tonne"
" within 18 months, a scenario that, she warned, would compel the IMF to invoke Article VII, Section 4(b) of its 1944 charter, "
"thereby obliging member nations to raise their collective green-bond contributions by 11 percent of GDP (roughly €2.7 trillion),"
" unless—God forbid—an unforeseen geo-engineering breakthrough achieving at least a 0.27 W m⁻² reduction in radiative forcing emerges"
" before the 22-nation summit set for 09:00 UTC on 5 May 2027 at Reykjavik’s Harpa Concert Hall. "
"Expected Output:"
"During the late afternoon of 14 October 2024, the International Association for Climate Economics (IACE) released its 312-page"
"report titled “Carbon Costs and the Tomorrow Market.”"
"During the same afternoon, Dr. María-José Fernández spoke to 27 delegates from 9 EU states."
"Dr. Fernández won the 2008 Intel Science Award at 16 years old."
"She now heads the quantum-finance lab at MIT."
"She said the projected “social cost of carbon” could climb above $173.45 per metric tonne within 18 months."
"This will happen if global CO₂ emissions exceed 56.3 gigatonnes by 2030."
"She warned that this scenario would compel the IMF to invoke Article VII, Section 4(b) of its 1944 charter."
"This will oblige member nations to raise their collective green-bond contributions by 11 percent of GDP."
"This is roughly €2.7 trillion."
"This obligation will apply unless an unforeseen geo-engineering breakthrough emerges."
"The breakthrough must achieve at least a 0.27 W m⁻² reduction in radiative forcing."
"The breakthrough must occur before the 22-nation summit."
"The summit is set for 09:00 UTC on 5 May 2027."
"The summit will take place at Reykjavik’s Harpa Concert Hall."
"Input sentence:"
"{text}")
    prompt = prompt_template.format(text=text)
    return model.invoke(prompt)

def simplify_text_stage2(text: str, model) -> str:
    prompt_template = PromptTemplate.from_template(
"Task: Split and format the input so every sentence"
"fully complies with the formatting rules for our NER pre-processor."
"STRICT RULES"
"1. One fact per sentence."
"2. No relative words: who, which, that, whose, where, when."
"3. No clause-joining conjunctions (and, but, because, so, unless, while, although, however) inside one sentence." 
"4. Conditionals / causals must be a two-step chain:  "
   "- Sentence A begins with “If …” and is a complete sentence that ends with a full stop."
   "- Sentence B begins with “Then …”, “This will …”, or “This would …”."
"5. Never invent new information. If a fix would drop a fact, copy the original line unchanged."
"6. Every sentence must be a grammatically complete, standalone sentence"
   "(subject + verb). No fragments such as “At 16 years old.” or “She said.”"
"7. A sentence may not start with a bare preposition (e.g. “At …”, “In …”) unless it also contains a subject and verb."
"8. Return only the final sentence(s), one per line. Do not show your reasoning."
"PROCESS "
"For each input line:"  
"- If the line already follows all rules, output it unchanged."  
"- Otherwise, split, rewrite, or reorder phrases just enough to satisfy"
"  every rule **without modifying factual content**."
"Input sentence:"
"{text}")
    prompt = prompt_template.format(text=text)
    return model.invoke(prompt)

def create_knowledge_ontology(text: str, model) -> str:
    prompt_template = PromptTemplate.from_template(
"""
    You are an information-extraction assistant.

    **Task**
    1. Read the sentence given in <text>.
    2. Identify every distinct entity (noun / noun-phrase) → create a node for each.
    3. Identify every *verb or preposition* that expresses a relation between two entities → create an edge.

    **Output format (MUST follow exactly – NO extra keys, NO prose):**
    {{
      "nodes": [
        {{"id": "n1", "label": "<entity-1>"}},
        ...
      ],
      "edges": [
        {{"source": "nX", "relation": "<predicate>", "target": "nY"}},
        ...
      ]
    }}

    *Rules*
    • Re-use a node if you see the same surface form again (case-insensitive).  
    • Use singular labels (“delegates”, “delegates' ” → “delegate”).  
    • relation = raw verb/preposition in **lower case**.  
    • If no relation is found, return an empty "edges" array.  
    • Do **NOT** output anything except the JSON block.

    Example
    Input: "Marie Curie discovered radium with Pierre Curie in 1898."
    Output:
    {{
        "nodes": [
            {{"id": "n1", "label": "marie curie"}},
            {{"id": "n2", "label": "pierre curie"}},
            {{"id": "n3", "label": "radium"}},
            {{"id": "n4", "label": "1898"}}
        ],
        "edges": [
            {{"source": "n1", "relation": "discovered", "target": "n3"}},
            {{"source": "n2", "relation": "discovered", "target": "n3"}},
            {{"source": "n1", "relation": "with", "target": "n2"}},
            {{"source": "n3", "relation": "in", "target": "n4"}}
        ]
    }}
    <text>{text}</text>"""
    )
    prompt = prompt_template.format(text=text)
    return model.invoke(prompt)

def remove_think_block(text: str) -> str:
    # Remove <think>...</think> including the tags
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

def fuse_atomic_graphs(full_sentence: str, atomic_graphs: list[dict], model):
    prompt_template = PromptTemplate.from_template(
        """
You are a knowledge-graph engineer.

# INPUTS
## Full_sentence
{full_sentence}

## Atomic_graphs
{atomic_graphs}

# TASK
Using *both* the Full_sentence and the Atomic_graphs:

1. **Merge nodes that refer to the same entity**  
   • Same name (case-insensitive) ➜ single node  
   • Pronouns (“she”, “it”, “this”) ➜ resolve to their antecedent entity  
   • Treat abbreviations & long forms as one (e.g. “IMF” = “International Monetary Fund”).

2. **Add missing relations that express logic in the original sentence**, especially:  
   • *causal*  → relation = "causes"  
   • *conditional* (“if … then …”) → relation = "condition_for"  
   • *exception* (“unless …”)   → relation = "exception_to"  
   • *temporal* (“before / after / during”) → relation = "temporal_relation" with
     an attribute "type": "before" | "after" | "during".

3. **Normalise relation vocabulary** to the following set  
   {located_in, occurs_on_date, heads, speaks_to, wins, exceeds, causes,
    condition_for, exception_to, raises, invokes, increases_to, achieves,
    temporal_relation}

4. **Convert literal quantities to node attributes** instead of separate nodes  
   • e.g. "€2.7 trillion" ➜ node "green-bond contribution" {amount: 2.7e12, currency: "EUR"}  
   • "0.27 W m⁻²" ➜ {value:0.27, unit:"W m^-2"}.

5. **Return strict JSON** exactly in this schema ­— nothing else:

```json
{
  "nodes":[
    {"id":"n1","label":"...", "...optional_attributes":...},
    ...
  ],
  "edges":[
    {"source":"nX","relation":"<one_of_allowed_relations>","target":"nY", "attributes":{...optional...}},
    ...
  ]
} """
"""Return a single ontology fragment covering the whole sentence.""")
    prompt = prompt_template.format(full_sentence=full_sentence,atomic_graphs=json.dumps(atomic_graphs, ensure_ascii=False, indent=2))
    return model.invoke(prompt)