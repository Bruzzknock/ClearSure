from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import re

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
"STRICT RULES  "
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
"Task: Take the input and create a simple ontology." \
"Output the node(s) and edge(s) of the ontology." \
"Nodes are represented: [node]'name of the node'" \
"Edges are represented: ->'name of the edge'" \
"Example Input: Dr. María-José Fernández spoke to 27 delegates during the same afternoon." \
"Example Output: [node]Dr. María-José Fernández ->spoke [node]27 delegates"
"Input:"
"{text}")
    prompt = prompt_template.format(text=text)
    return model.invoke(prompt)

def remove_think_block(text: str) -> str:
    # Remove <think>...</think> including the tags
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)