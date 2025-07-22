# ClearSure
ClearSure is an open-source AI built for context-aware, definitive insights like in insurance, finance, and law. Harnessing advanced NLP, it delivers accurate, trustworthy answers you can rely on for high-stakes decisions. Join our community to refine ClearSure’s capabilities.


# Preperation
1. change requirements.in - update packages
2. pip-compile --generate-hashes requirements.in - this will create requirements.txt
3. pip install -r requirements.txt

# Some binaries have to be manually installed
1. powershell choco install poppler
2. powershell choco install tesseract

## Topic extraction
After processing all sentences, the pipeline derives a concise topic label from
the document’s statements. The topic node (`t#`) is linked to the first
statement via the relation `BELONGS_TO_TOPIC`.
