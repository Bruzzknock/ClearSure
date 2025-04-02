from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from extractor.semantic_node import SemanticNode

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
segmenter = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def summarize_node_text(text):
    return summarizer(text, max_length=100, min_length=20, do_sample=False)[0]["summary_text"]

# Step 1: Get 1–3 word summary
def get_short_title(text):
    prompt = "Give a 1–3 word title describing the topic of this text."
    result = segmenter(f"{prompt}\n\n{text}", max_length=10)[0]["generated_text"]
    return result.strip().replace(".", "")

# Step 2: Top-level segmentation prompt
def get_top_sections(text):
    prompt = (
        "Divide the following document into top-level sections by meaning, "
        "not by formatting. List them by title. Make them readable by spliting them with ====. Example: AUTO INSURANCE POLICY ==== Policy Overview ==== Coverage Details"
    )
    result = segmenter(f"{prompt}\n\n{text}", max_length=256)[0]["generated_text"]
    print("!!!!!!!!!!!!!!!!!!!!!!!SECTIONS RESULT!!!!!!!!!!!!!!!!!!!!!!!!!!", result)
    section_titles = [line.strip("-• \n") for line in result.split("\n") if line.strip()]
    return section_titles

# Step 3: Segment by section content
def extract_section_bodies(text, section_titles):
    # Use LLM again or use simple embedding similarity to group content under titles
    # For now, naively chunk by order
    chunks = []
    for i in range(len(section_titles)):
        title = section_titles[i]
        if i < len(section_titles) - 1:
            part = text.split(section_titles[i + 1])[0].split(title)[-1]
        else:
            part = text.split(title)[-1]
        chunks.append((title, part.strip()))
    return chunks

# Recursive generator
def generate_node_hierarchy_semantic(text, level=0, title=None):
    if level == 0:
        title = get_short_title(text)

    node = SemanticNode(title, level, text)
    # node.summary = summarize_node_text(text)

    if level < 2:  # Limit depth for now
        try:
            section_titles = get_top_sections(text)
            sections = extract_section_bodies(text, section_titles)
            for sec_title, sec_text in sections:
                child = generate_node_hierarchy_semantic(sec_text, level=level+1, title=sec_title)
                node.children.append(child)
        except Exception as e:
            print(f"Sectioning failed at level {level}: {e}")

    return node
