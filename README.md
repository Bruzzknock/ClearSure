# ClearSure
ClearSure is an open-source AI built for context-aware, definitive insights like in insurance, finance, and law. Harnessing advanced NLP, it delivers accurate, trustworthy answers you can rely on for high-stakes decisions. Join our community to refine ClearSure’s capabilities.


# Preperation
1. change requirements.in - update packages
2. pip-compile --generate-hashes requirements.in - this will create requirements.txt
3. pip install -r requirements.txt

# Some binaries have to be manually installed
1. powershell choco install poppler
2. powershell choco install tesseract

## Knowledge Graph Architecture

Documents are ingested and converted to a two-layer knowledge graph:

1. **Statement layer** – Every sentence-level fact is preserved as a
   `:Statement` node. This forms an immutable audit trail of the source text.

2. **Company view** – Cleaned entities and their relations build the
   operational view of the "brain". Logical rules are explicit `:Rule` nodes
   connected to statements via `:HAS_CONDITION` and `:HAS_CONCLUSION` edges.

3. **Topic layer** – After processing all statements, related sentences are
   summarised into short topics (`t#`). Statement nodes link to these topics via
   `BELONGS_TO_TOPIC`, providing a lightweight navigation hierarchy.

This design keeps logical operators out of entity space while making it easy to
traverse from conditions to conclusions.