from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import re
import json

def simplify_text(text: str, model) -> str:
    prompt_template = PromptTemplate.from_template(
"""
############################################################
# ATOMIC FACT & LOGIC-MAP EXTRACTOR #
############################################################

Return a numbered list of atomic statements plus logic links.

FORMAT
• One line per statement, order = first appearance in text  
• Prefix lines with  sN -  (N = 1, 2, 3 …)  
• Prefix time-anchors with      wN -   (independent counter).  
• Cross-reference any item with [sN] or [wN].  
• No blank lines or commentary.

STATEMENT TYPES
1. **Fact** … “X did Y / is Y / has Y.”  
   – If the sentence contains an explicit time phrase:  
        · If the phrase has **3 or more tokens**, extract it as a new wN  
          and append  `[WHEN = wN]`  to the fact.  
        · If it is **1- or 2-tokens**, keep it inline:  
          `… [WHEN = tomorrow]`, `… [WHEN = now]`, etc. 
2. Alias … “NASA is the National Aeronautics and Space Administration.”  
3. Quantity … keep number + unit + noun (“56 Gt CO₂”).  
4. Simple conditional … IF [cond] THEN [cons].  
5. **Boolean conditional** … `IF (<expr>) THEN [cons]`, where  
   <expr> ::= [sN] | (NOT <expr>) | (<expr> AND/OR <expr>)  
          | BEFORE [wN/sN] | AFTER [wN/sN] | AT [wN/sN]

RULES
• Split until each line conveys **exactly one** idea.  
• Keep numbers with context; stand-alone units are OK when nouns.  
• When a short form & long form first co-occur, add an Alias line.  
• No bare verbs or causal fragments as stand-alone statements.  
• Verify that the s-index and w-index each run 1…N with no gaps.

MINI EXAMPLES
Input A: “Ada Lovelace waited seven minutes before firing at 1 kHz.”  
Output A:  
s1 - Ada Lovelace waited seven minutes.  
s2 - Ada Lovelace fired at 1kHz.  
s3 - IF [s1] THEN [s2].

Input B:  
“At 09:00 UTC, if sensor A is offline **or** NOT sensor B is calibrated, the controller will enter safe-mode.”  
Output B:  
w1 - at 09:00 UTC  
s1 - sensor A is offline.  
s2 - sensor B is calibrated.
s3 - the controller enters safe-mode
s4 - IF ((AT [w1] AND [s1]) OR NOT [s2]) THEN [s3].


Input C:
“If the sensor is offline until 31 Dec 2025 or it ever fails after that date, the controller enters safe-mode during 
the inspection window of 1–15 Jan 2026.”
Output C:
s1 - the sensor is offline [WHEN = [w1]].
w1 - until 31 December 2025
s2 - the sensor fails.
s3 - the controller enters safe-mode [WHEN = [w2]].
w2 - during the inspection window of 1–15 Jan 2026.
s4 - IF ([s1] OR (AFTER [w1] AND [s2])) THEN [s3].

FULL EXAMPLE:
Input: 
Throughout the late afternoon of 14 October 2024—after the International Association for Climate Economics (IACE)
released its 312-page report titled “Carbon Costs and the Tomorrow Market”—Dr. María-José Fernández, who at 16 years old won
the 2008 Intel Science Award and now heads the quantum-finance lab at MIT, told 27 delegates from 9 EU states that if global
CO₂ emissions exceed 56.3 gigatonnes by 2030, the projected “social cost of carbon” could climb above $173.45 per metric tonne
within 18 months, a scenario that, she warned, would compel the IMF to invoke Article VII, Section 4(b) of its 1944 charter, 
thereby obliging member nations to raise their collective green-bond contributions by 11 percent of GDP (roughly €2.7 trillion),
unless—God forbid—an unforeseen geo-engineering breakthrough achieving at least a 0.27 W m⁻² reduction in radiative forcing emerges
before the 22-nation summit set for 09:00 UTC on 5 May 2027 at Reykjavik’s Harpa Concert Hall. 
Output:
w1 - Throughout the late afternoon of 14 October 2024.
s1 - The International Association for Climate Economics released its 312-page report.
s2 - IACE is the International Association for Climate Economics.
s3 - The 312-page report is titled “Carbon Costs and the Tomorrow Market”.
w2 - at 16 years old.
s4 - Dr. María-José Fernández won the 2008 Intel Science Award [WHEN = w2].
s5 - Dr. María-José Fernández heads the quantum-finance lab at MIT [WHEN = now].
s6 - Dr. María-José Fernández told 27 delegates [WHEN = [s8]].
s7 - 27 delegates are from 9 EU states.
s8 - IF (AT [w1] AND AFTER [s2]) THEN [s6].
s9 - Global CO₂ emissions exceed 56.3 gigatonnes [WHEN = by 2030].
s10 - The projected social cost of carbon climbs above $173.45 per metric tonne [WHEN = [w3]].
w3 - within 18 months.
s11 - IF [s9] THEN [s10].
s12 - The IMF invokes Article VII, Section 4(b) of its 1944 charter.
s13 - Member nations raise their collective green-bond contributions by 11 percent of GDP.
s14 - IF [s11] THEN [s12].
s15 - IF [s12] THEN [s13].
s16 - 11 percent of GDP is roughly €2.7 trillion.
s17 - An unforeseen geo-engineering breakthrough achieves at least 0.27 W m⁻² reduction in radiative forcing.
s18 - A 22-nation summit is set at Reykjavik’s Harpa Concert Hall [WHEN = [w4]].
w4 - for 09:00 UTC on 5 May 2027
s19 - IF (BEFORE [s18] AND [s17]) THEN NOT [s15].

DATA  
<input>  
{input}  
</input>
""")
    prompt = prompt_template.format(input=text)
    return model.invoke(prompt)


def extract_entities(text: str, model) -> str:
    prompt_template = PromptTemplate.from_template(
        """
### TASK
Extract every **reference entity** from the text between <input></input>.

A reference entity = any stand-alone noun phrase (person, organisation, place,
time or date, quantity, unit, title, duration, group, acronym, etc.).
If uncertain if something should be an entitiy or not, better to include it.
*Do NOT extract verbs, actions, or causal phrases (e.g. “trigger”, “if … then …”, “would compel”).*

### OUTPUT
• One line per entity, in order of first appearance  
• Format:  eN - <verbatim text>   (N = 1, 2, 3 …)  
• No commentary, no blank lines

### RULES
1. Long-form and acronym/alias are kept as **separate** entities (same order).  
2. Keep numbers **with** their units *and* the object they quantify  
   («14.8 billion Ɡ», «1 000 m depth», «−0.15 W m⁻²»).  
3. Counts and relative durations are entities («43 parliamentarians»,
   «seven minutes before», «within 18 months»).  
4. Stand-alone units/symbols («MW», «%», «UTC») are entities when they
   appear as nouns on their own.    
5. Do **not** extract verbs, verb phrases, or whole clauses that convey
   actions, conditions, or causation.  
6. At the end, verify that indices run 1…N with no gaps or duplicates.

### MICRO EXAMPLE
Input:  “Sir Ada Lovelace, waited seven minutes before launching at 1 kHz.”
Output:
e1 - Sir Ada Lovelace    
e2 - seven minutes  
e3 - 1 kHz

### FULL EXAMPLE
Input:
Throughout the late afternoon of 14 October 2024—after the International Association for Climate Economics (IACE) 
released its 312-page report titled “Carbon Costs and the Tomorrow Market”—Dr. María-José Fernández, who at 16 years 
old won the 2008 Intel Science Award and now heads the quantum-finance lab at MIT, told 27 delegates from 9 EU states 
that if global CO₂ emissions exceed 56.3 gigatonnes by 2030, the projected “social cost of carbon” could climb above 
$173.45 per metric tonne within 18 months, a scenario that, she warned, would compel the IMF to invoke Article VII, 
Section 4(b) of its 1944 charter, thereby obliging member nations to raise their collective green-bond contributions 
by 11 percent of GDP (roughly €2.7 trillion), unless—God forbid—an unforeseen geo-engineering breakthrough achieving 
at least a 0.27 W m⁻² reduction in radiative forcing emerges before the 22-nation summit set for 09:00 UTC on 5 May 2027 
at Reykjavik’s Harpa Concert Hall.

Output:
e1 - throughout the late afternoon of 14 October 2024
e2 - the International Association for Climate Economics
e3 - IACE
e4 - 312-page report
e5 - “Carbon Costs and the Tomorrow Market”
e6 - Dr. María-José Fernández
e7 - at 16 years old
e8 - the 2008 Intel Science Award
e9 - quantum-finance lab
e10 - MIT
e11 - 27 delegates 
e12 - 9 EU states
e13 - global CO₂ emissions
e14 - 56.3 gigatonnes
e15 - 2030
e16 - the projected “social cost of carbon”
e17 - $173.45 
e18 - metric tonne
e19 - 18 months
e20 - the IMF
e21 - Article VII, Section 4(b) of its 1944 charter
e22 - member nations
e23 - collective green-bond contributions
e24 - 11 percent of GDP
e25 - roughly €2.7 trillion
e26 - unforeseen geo-engineering breakthrough
e27 - 0.27 W m⁻² reduction in radiative forcing
e28 - the 22-nation summit
e29 - 09:00 UTC on 5 May 2027
e30 - Reykjavik’s Harpa Concert Hall
</example>
<input>{input}</input>
        """
    )
    prompt = prompt_template.format(input=text)
    return model.invoke(prompt)

def simplify_text_stage2(text: str, entities, model) -> str:
    prompt_template = PromptTemplate.from_template(
        """
######################################################################
#  ENTITY-QC  –  *Missing-Only pass*                                #
#  Caller provides two blocks:                                       #
#    <source></source>   – original passage                          #
#    <entities></entities> – step-1 output (eN – …)                  #
######################################################################

### TASK
Scan <source> for **reference entities** = any stand-alone noun phrase (see RULES).  
Output **only those entities that do NOT appear** anywhere in <entities>.

### OUTPUT
• One line per *missing* entity, in order of first appearance  
• Format:  eM - <verbatim text>   (M = 1, 2, 3 …)  
• No commentary.

### RULES  (brief)
• Reference entity = any stand-alone noun phrase: person, organisation, place, date/time, quantity, unit, title, duration, group, acronym, etc.  
• Long form and acronym/alias are **separate** entities, recorded when each first appears.    
• Counts and relative durations are entities (“43 parliamentarians”, “seven minutes before”).  
• Do **NOT** include verbs, actions, causal links, or whole clauses.  
• Do **NOT** repeat anything already present (verbatim) in <entities>.  

### EXAMPLE
<example>
<source>
Alice met Bob in Paris.
</source>

<entities>
e1 - Alice
</entities>

OUTPUT MISSING ENTITIES:
e1 - Bob
e2 - Paris
</example>

### DATA
<source>
{SOURCE_TEXT}
</source>

<entities>
{ENTITY_LIST}
</entities>

### OUTPUT
e1 - …
e2 - …
…


"""
)
    prompt = prompt_template.format(SOURCE_TEXT=text, ENTITY_LIST=entities)
    return model.invoke(prompt)

def create_knowledge_ontology(text: str, model) -> str:
    prompt_template = PromptTemplate.from_template(
"""
############################################################
# FACT-BLOCK → KNOWLEDGE-GRAPH JSON #
############################################################

You will receive a single  <FACTS> … </FACTS>  block that follows the
“Atomic Fact & Logic-Map Extractor” format (sN / wN lines, optional
IF-THEN statements, etc.).

Return **exactly one** JSON object shaped like

{{
  "nodes": [ … ],
  "edges": [ … ]
}}

No extra keys, no commentary, no ''' json, and no blank lines outside the JSON.

-----------------------------------------------------------------
NODE-RULES
-----------------------------------------------------------------
1. **Entity nodes**  
   • Create one node per distinct noun-phrase that plays a subject or
     object role.  
   • Shape:  {{ "id":"nK", "label":"<text>" }}

2. **Time-anchor nodes** (all wN lines)  
   • Shape:  {{ "id":"wN", "label":"<full time phrase>" }}

3. **Statement surrogate nodes**  
   • One per Fact (not for Alias / Quantity / IF-THEN).  
   • Copy the full Fact text into  "label".  
   • Add  "type":"Statement"  and  "edgeId":"e_sN".  
   • Shape:  {{ "id":"sN", "label":"<fact text>", "type":"Statement",
               "edgeId":"e_sN" }}

-----------------------------------------------------------------
EDGE-RULES
-----------------------------------------------------------------
A. **Original verb edge** (for each Fact)  
   • Between subject-node and object-node.  
   • Normalize the verb to snake_case for  "relation".  
   • Add  "edgeId":"e_sN".  
   • Shape:  {{ "source":"nX", "relation":"<verb>", "target":"nY",
               "edgeId":"e_sN" }}

B. **Role edges** tying entities to their Statement surrogate  
   • {{ "source":"<subject-node>", "relation":"ACTOR_IN", "target":"sN" }}  
   • {{ "source":"<object-node>", "relation":"OBJECT_IN", "target":"sN" }}

C. **WHEN edges**  
   • For any “… [WHEN = wN]” tag attach  
     {{ "source":"sN", "relation":"WHEN", "target":"wN" }}

D. **Logic edges** from IF / THEN lines  
   • Use  "relation":"CAUSES"  
   • Example  
     {{ "source":"s3", "relation":"CAUSES", "target":"s5" }}

-----------------------------------------------------------------
NAMING & CONSISTENCY
-----------------------------------------------------------------
•  edgeId is deterministic →  "e_s1", "e_s2", …  
•  Don’t create duplicate nodes; reuse them when the text matches.  
•  The **nodes** array first, then the **edges** array.  
•  Output valid JSON, nothing else.

-----------------------------------------------------------------
EXAMPLE (input trimmed)

<FACTS>
s1 - Ada Lovelace waited seven minutes.
s2 - Ada Lovelace fired at 1 kHz.
s3 - IF [s1] THEN [s2].
w1 - at 09:00 UTC
s4 - the test occurs [WHEN = w1].
</FACTS>

⇒ Expected shape (spacing irrelevant)

{{
  "nodes":[
    {{"id":"n1","label":"Ada Lovelace"}},
    {{"id":"n2","label":"seven minutes"}},
    {{"id":"n3","label":"1 kHz"}},
    {{"id":"n4","label":"the test"}},
    {{"id":"w1","label":"at 09:00 UTC"}},
    {{"id":"s1","label":"Ada Lovelace waited seven minutes.","type":"Statement","edgeId":"e_s1"}},
    {{"id":"s2","label":"Ada Lovelace fired at 1 kHz.","type":"Statement","edgeId":"e_s2"}},
    {{"id":"s4","label":"the test occurs","type":"Statement","edgeId":"e_s4"}}
  ],
  "edges":[
    {{"source":"n1","relation":"waited","target":"n2","edgeId":"e_s1"}},
    {{"source":"n1","relation":"ACTOR_IN","target":"s1"}},
    {{"source":"n2","relation":"OBJECT_IN","target":"s1"}},

    {{"source":"n1","relation":"fired_at","target":"n3","edgeId":"e_s2"}},
    {{"source":"n1","relation":"ACTOR_IN","target":"s2"}},
    {{"source":"n3","relation":"OBJECT_IN","target":"s2"}},

    {{"source":"n4","relation":"occurs","target":"w1","edgeId":"e_s4"}},
    {{"source":"n4","relation":"ACTOR_IN","target":"s4"}},
    {{"source":"w1","relation":"OBJECT_IN","target":"s4"}},
    {{"source":"s4","relation":"WHEN","target":"w1"}},

    {{"source":"s1","relation":"CAUSES","target":"s2"}}
  ]
}}

############################################################
# END OF PROMPT #
############################################################


<FACTS>{facts}</FACTS>
""")
    prompt = prompt_template.format(facts=text)
    return model.invoke(prompt)

def remove_think_block(text: str) -> str:
    # Remove <think>...</think> including the tags
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

def fuse_atomic_graphs(full_sentence: str, atomic_graphs: list[dict], model):
    prompt_template = PromptTemplate.from_template(
        """
You are a knowledge-graph synthesis assistant.

────────────────────────────────  INPUT  ───────────────────────────────
 <INPUT_TEXT>                                                            
 {original}                                                            
 </INPUT_TEXT>                                                           
                                                                       
 <ATOMIC_GRAPHS>                                                              
 # a JSON list where each item is a mini-graph with "nodes"/"edges"    
 # produced from simplified sentences                                                              
 {atomic}                                                              
 </ATOMIC_GRAPHS>                                                             

──────────────────────────  TASK  ─────────────────────────────────────
 Build **one unified knowledge graph** that captures *every* entity,   
 attribute, and relationship expressed in the original sentence.  Use  
 the atomic graphs only as hints; correct or complete them whenever    
 the original sentence has information they missed.                    
                                                                       
 Return exactly one JSON object with this schema — nothing else:       
                                                                       
 {{                                                                     
   "nodes": [                                                          
     {{"id": "n1", "label": "<canonical label>", attributes": "…optional…"}}, 
     …                                                                 
   ],                                                                  
   "edges": [                                                          
     {{"source": "nX", "relation": "<one_of_allowed_relations>",        
      "target": "nY", "attributes": "…optional…"}},                     
     …                                                                 
   ]                                                                   
 }}                                                                     

──────────────  NODE RULES  ──────────────
 - One node per *distinct* real-world     
   entity or literal value.               
 - Canonical label: lower-case            
   (keep ALL-CAP acronyms as is), strip   
   leading articles (“the”, “a”, “an”),   
   keep internal spaces, no underscores.  
 - Create a second node for any acronym   
   in parentheses and connect it to the   
   full form with relation `"is"`.        
 - Optional node attributes you may add:  
   "type", "value", "unit", "date",       
   "quantity", "role".                    

──────────────  EDGE RULES  ──────────────
 - Add an edge for every *explicit*       
   grammatical or logical link:           
     – main verbs (released, spoke …)     
     – copula / alias (is, was, are)      
     – functional verbs (titled, heads…)  
     – prepositions (to, of, by, at …)    
     – logical connectors (if, unless,    
       before, after, then, causes,       
       compels, obliges, within, by,      
       exceed, above, per, achieve…)      
 - Relation text = **exact token(s)**     
   from the sentence, with spaces (never  
   underscores).  Lower-case verbs; keep  
   acronyms upper-case.                   
 - Every `source` and `target` id must    
   exist in `"nodes"`.                    
 - For conditional logic use edges like   
       nX —if→ nY     nX —unless→ nY      
   and chain further consequences with    
       nY —then→ nZ                       
 - For rate / amount phrases link with    
   the introducing preposition            
   (nPrice —per→ nUnit).                  
 - Optional edge attributes allowed, e.g. 
   {{"modal":"could"}}, {{"tense":"past"}}. 

────────────── COVERAGE CHECK ────────────
 Before you answer, run this mental list: 
 - Every node/edge from <ATOMIC_GRAPHS> is      
    represented (duplicates merged).      
 - Every extra fact in <original> that   
    is NOT in <ATOMIC_GRAPHS> is added.          
 - No pronouns (“she”, “it”, “this”)     
    remain as labels.                     
 - No dangling ids (all edges refer to   
    ids present in "nodes").              
 - No markdown, no explanation text.

Return the final graph JSON **and nothing else**.
""")
    prompt = prompt_template.format(original=full_sentence,atomic=json.dumps(atomic_graphs, ensure_ascii=False, indent=2))
    return model.invoke(prompt)