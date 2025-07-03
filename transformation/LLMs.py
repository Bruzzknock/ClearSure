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

──────────────────── TASK ────────────────────
Given **one sentence** (inside <text>) build a mini knowledge-graph that
captures every explicit entity and every explicit relation in that sentence
only.  Output a single JSON object that matches the schema shown below.

Entities → *nodes*  
Relations → *edges*

────────────── How to build the graph ──────────────
NODES  
• Create one node for every noun-phrase that denotes a person, organisation,
  object, document, concept, number, money amount, unit, date, or title that
  appears in the sentence.  
• Keep the surface text exactly as it appears (lower-case words, internal
  capitals for acronyms, preserve numerals, hyphens, currency symbols).  
• When a parenthetical acronym follows a full name, **create a second node for
  the acronym**.  
  – Immediately connect it to the full form with the edge
    `{{"source": "<acronym-id>", "relation": "is", "target": "<full-name-id>"}}`.  
• Assign ids `n1 … nK` in order of first appearance.
• Treat every explicit date, time-of-day, duration, location or rate phrase as a 
node and link it with the introducing preposition (‘on’, ‘during’, ‘at’, ‘per’, …).

EDGES  
• For every explicit grammatical link between two entities, add an edge:  
    – Main verbs (`released`, `spoke`, `won`, `heads`, `located`, etc.).  
    – Copula/alias links (`is`, `was`, `are`).  
    – Functional verbs such as `titled`, `scheduled`, `oblige`.  
    – Prepositions that attach one entity to another (`to`, `of`, `by`, `at`, …).  
• Relation text = the word *exactly as it appears* in the sentence,
  lower-cased; use underscores for multi-word verbs if needed (`spoke_to`).  
• Do not invent relations that are not present.

OUTPUT (must match this shape exactly – no extra keys, no prose)
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

────────────── Example ──────────────
Input:
During the late afternoon of 14 October 2024, the International 
Association for Climate Economics (IACE) released its 312-page report 
titled “Carbon Costs and the Tomorrow Market.”

Output:
{{
  "nodes": [
    {{"id": "n1", "label": "international association for climate economics"}},
    {{"id": "n2", "label": "iace"}},
    {{"id": "n3", "label": "312-page report"}},
    {{"id": "n4", "label": "carbon costs and the tomorrow market"}}

  ],
  "edges": [
    {{"source": "n1", "relation": "released", "target": "n3"}},
    {{"source": "n2", "relation": "is", "target": "n1"}},
    {{"source": "n3", "relation": "titled", "target": "n4"}}

  ]
}}

<text>{text}</text>
""")
    prompt = prompt_template.format(text=text)
    return model.invoke(prompt)

def remove_think_block(text: str) -> str:
    # Remove <think>...</think> including the tags
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

def fuse_atomic_graphs(full_sentence: str, atomic_graphs: list[dict], model):
    prompt_template = PromptTemplate.from_template(
        """
You are a knowledge-graph synthesis assistant.

────────────────────────────────  INPUT  ───────────────────────────────
 <original>                                                            
 {original}                                                            
 </original>                                                           
                                                                       
 <atomic>                                                              
 # a JSON list where each item is a mini-graph with "nodes"/"edges"    
 # produced from simplified sentences                                  
 # (the content of atomic_sentences.json)                              
 {atomic}                                                              
 </atomic>                                                             
────────────────────────────────────────────────────────────────────────


──────────────────────────  TASK  ─────────────────────────────────────
 Build **one unified knowledge graph** that captures *every* entity,   
 attribute, and relationship expressed in the original sentence.  Use  
 the atomic graphs only as hints; correct or complete them whenever    
 the original sentence has information they missed.                    
                                                                       
 Return exactly one JSON object with this schema — nothing else:       
                                                                       
 {{                                                                     
   "nodes": [                                                          
     {{"id": "n1", "label": "<canonical label>", …optional_attributes}}, 
     …                                                                 
   ],                                                                  
   "edges": [                                                          
     {{"source": "nX", "relation": "<one_of_allowed_relations>",        
      "target": "nY", "attributes": "…optional…"}},                     
     …                                                                 
   ]                                                                   
 }}                                                                     

──────────────  NODE RULES  ──────────────
 • One node per *distinct* real-world     
   entity or literal value.               
 • Canonical label: lower-case            
   (keep ALL-CAP acronyms as is), strip   
   leading articles (“the”, “a”, “an”),   
   keep internal spaces, no underscores.  
 • Create a second node for any acronym   
   in parentheses and connect it to the   
   full form with relation `"is"`.        
 • Optional node attributes you may add:  
   "type", "value", "unit", "date",       
   "quantity", "role".                    

──────────────  EDGE RULES  ──────────────
 • Add an edge for every *explicit*       
   grammatical or logical link:           
     – main verbs (released, spoke …)     
     – copula / alias (is, was, are)      
     – functional verbs (titled, heads…)  
     – prepositions (to, of, by, at …)    
     – logical connectors (if, unless,    
       before, after, then, causes,       
       compels, obliges, within, by,      
       exceed, above, per, achieve…)      
 • Relation text = **exact token(s)**     
   from the sentence, with spaces (never  
   underscores).  Lower-case verbs; keep  
   acronyms upper-case.                   
 • Every `source` and `target` id must    
   exist in `"nodes"`.                    
 • For conditional logic use edges like   
       nX —if→ nY     nX —unless→ nY      
   and chain further consequences with    
       nY —then→ nZ                       
 • For rate / amount phrases link with    
   the introducing preposition            
   (nPrice —per→ nUnit).                  
 • Optional edge attributes allowed, e.g. 
   {{"modal":"could"}}, {{"tense":"past"}}. 

────────────── COVERAGE CHECK ────────────
 Before you answer, run this mental list: 
 ✅ Every node/edge from <atomic> is      
    represented (duplicates merged).      
 ✅ Every extra fact in <original> that   
    is NOT in <atomic> is added.          
 ✅ No pronouns (“she”, “it”, “this”)     
    remain as labels.                     
 ✅ No dangling ids (all edges refer to   
    ids present in "nodes").              
 ✅ No markdown, no explanation text.

Return the final graph JSON **and nothing else**.
""")
    prompt = prompt_template.format(original=full_sentence,atomic=json.dumps(atomic_graphs, ensure_ascii=False, indent=2))
    return model.invoke(prompt)