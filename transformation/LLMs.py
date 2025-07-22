from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import re
import json
from typing import List
from kg_utils import _extract_json_block

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

4. **Rule nodes** for IF/THEN lines
   • Shape:  {{ "id":"rN", "label":"<rule text>", "type":"Rule" }}

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

D. **Rule edges** linking Rule nodes to Statements
   • For each IF/THEN line create one Rule node rN.
   • Connect it with {{ "source":"rN", "relation":"HAS_CONDITION", "target":"sX" }}
     for every condition statement, and
     {{ "source":"rN", "relation":"HAS_CONCLUSION", "target":"sY" }}
     for every conclusion statement.

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
    {{"id":"s4","label":"the test occurs","type":"Statement","edgeId":"e_s4"}},
    {{"id":"r1","label":"IF [s1] THEN [s2]","type":"Rule"}}
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

    {{"source":"r1","relation":"HAS_CONDITION","target":"s1"}},
    {{"source":"r1","relation":"HAS_CONCLUSION","target":"s2"}}
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

def clean_up_1st_phase(text: str, model):
    prompt_template = PromptTemplate.from_template(
        """
############################################################
# JSON-GRAPH CLEAN-UP  ➜  ADD-MISSING EDGES
############################################################

You receive **one** JSON object (the graph produced in pass #1).

Return **exactly one** JSON object shaped like

{{
  "edges_patch":[ … ]
}}

No other keys, no comments, no ```json fences.

─────────────────────────────────────────────────────────────
EDGE-GENERATION RULES
─────────────────────────────────────────────────────────────
1.  Temporal references inside IF / WHEN clauses  
    • Scan every Statement node whose *label* starts with **IF**.  
    • Patterns:  BEFORE [wN] AFTER [wN] AT … [wN]  
      – Create {{ "source":"sK","relation":"BEFORE","target":"wN" }}  
         or AFTER / AT (use the word that appears).  

2.  Nested time-anchors (w-node → w-node)  
    • If the *label* of a wK node contains “after [wX]” or “before [wX]”  
      – Create {{ "source":"wK","relation":"AFTER","target":"wX" }}  
         or BEFORE accordingly.  

3. IF-surrogate linking 
   • For each Statement node sK whose *label* matches  
     `IF (*[sA]*) THEN (*[sB]*)` (case-insensitive):  
       – Add {{ "source":"sA","relation":"ACTOR_IN","target":"sK" }}  
       – Add {{ "source":"sB","relation":"OBJECT_IN","target":"sK" }}

4.  **No duplicates** – skip an edge if the identical triple already
    exists in the original edges array.

5.  Do **not** invent edgeId for these patch edges.

─────────────────────────────────────────────────────────────
OUTPUT CHECKLIST  (remove before running)
✓  Only the key "edges_patch"?  
✓  Every item follows  source / relation / target  schema?  
✓  Relation spelling matches BEFORE / AFTER / AT?  
✓  JSON is the only thing in the reply?

############################################################
<input>
{input}
</input>
""")
    prompt = prompt_template.format(input=text)
    return model.invoke(prompt)

def one_sentence_summary(text: str, model) -> str:
    """
    Ask the LLM to rewrite *text* as a single, information-complete sentence.

    Returns the raw sentence (stripped).  Raises ValueError if the reply
    contains line-breaks or multiple sentences.
    """
    prompt_template = PromptTemplate.from_template(
        """
############################################################
# ONE-SENTENCE COMPREHENSIVE SUMMARY
############################################################
Return **exactly one** English sentence that captures every distinct
fact and nuance in the <INPUT> block.

RULES
• One single sentence.  No lists, no line breaks, no commentary.
• You MAY use conjunctions / clauses to keep everything in.
• Do NOT omit any key data (names, quantities, dates, causal links).

<INPUT>
{text}
</INPUT>
"""
    )
    prompt = prompt_template.format(text=text)
    reply = model.invoke(prompt).strip()
    reply  = remove_think_block(reply).strip()

    # quick heuristic: ensure only one sentence (no multiple periods)
    if reply.count(".") > 1:
        raise ValueError(f"Expected one sentence, got: {reply}")
    return reply


def propose_split_spans(
    text: str,
    model,
) -> List[dict]:
    """
    Ask the LLM to suggest 2–max_segments non-overlapping character-offset
    spans that together cover *text*.

    The model must return:
      { "spans": [ { "start": int, "end": int }, … ] }

    *If the model uses half-open slices [start,end), they are converted to
    inclusive end indices.*
    """
    prompt_template = PromptTemplate.from_template(
        """
SYSTEM
You are “DeepSegment-2”.  
Return only raw JSON; stop after you output it.

USER
############################################
#  TWO-SPAN TEXT SPLITTER  →  JSON OUTPUT  #
############################################

▼ GOAL  
Create **two** coherent segments that together cover **all** characters in <INPUT>.

▼ HOW TO CHOOSE THE SINGLE SPLIT  
1. Scan top-to-bottom for the **first strong topic shift**, such as:  
   • a new top-level heading (numbered or unnumbered)  
   • a section whose purpose clearly differs from the preceding one  
   • a blank-line gap followed by a heading-style line (Title Case, ALL CAPS, or leading numbers like “2.”, “1.2”, etc.)  
2. Place the split **immediately before** that line’s first character.  
3. If the whole text stays on one topic, fall back to the **closest sentence boundary to the 50 % mark**.  
4. You must always return two spans; never merge them into one.

▼ OUTPUT FORMAT (raw JSON)  
{{
  "spans": [
    {{"start": <int>, "end": <int>}},   // Segment 1: from char 0 to split-1
    {{"start": <int>, "end": <int>}}    // Segment 2: from split to last char
  ]
}}

▼ RULES (check each ✅)  
- ✅ Exactly two objects inside "spans".  
- ✅ 0-based offsets; `end` is **inclusive**.  
- ✅ No gaps, no overlaps.  
- ✅ No other keys, no commentary, no Markdown fences.  
- ✅ After printing the JSON object, stop.

<INPUT>
{text}
</INPUT>

"""
    )
    prompt = prompt_template.format(text=text)
    raw_reply = model.invoke(prompt).strip()
    reply = remove_think_block(raw_reply)
    reply = _extract_json_block(reply)

    # -------- parse ------------------------------------------------------
    try:
        spans = json.loads(reply)["spans"]
    except Exception as e:
        raise ValueError(f"LLM did not return valid JSON:\n{reply}") from e

    # -------- normalise half-open slices --------------------------------
    if spans and spans[-1]["end"] == len(text):
        for s in spans:
            s["end"] -= 1                          # make 'end' inclusive

    # -------- sanity checks ---------------------------------------------
    if not isinstance(spans, list) or len(spans) < 2:
        raise ValueError(f"Need ≥2 spans, got: {spans}")
    
    print("SPANS: ", spans)
    print("LENGTH TEXT: ", len(text))

    prev_end = -1
    for span in spans:
        if {"start", "end"} - span.keys():
            raise ValueError("Each span must have start & end keys.")
        if span["start"] != prev_end + 1:
            raise ValueError("Spans have a gap or are out of order.")
        if span["start"] > span["end"]:
            raise ValueError("start > end in span.")
        prev_end = span["end"]

    if prev_end != len(text) - 1:
        raise ValueError("Spans do not cover entire text.")

    return spans


def label_text(text: str, model) -> str:
    """Return a short label describing *text*."""
    prompt_template = PromptTemplate.from_template(
        "Give a concise node label (<= 12 words) describing the following text:\n\n{text}"
    )
    prompt = prompt_template.format(text=text)
    return model.invoke(prompt)


def sentence_topic_same(topic: str, sentence: str, model) -> bool:
    """Determine whether *sentence* elaborates on *topic*."""
    prompt_template = PromptTemplate.from_template(
        "Topic: {topic}\nSentence: {sentence}\nDoes the sentence elaborate on this topic? Answer yes or no."
    )
    prompt = prompt_template.format(topic=topic, sentence=sentence)
    reply = model.invoke(prompt)
    return reply.lower().startswith("yes")
