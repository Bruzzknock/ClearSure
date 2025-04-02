from transformers import pipeline

class SemanticNode:
    def __init__(self, title, level, text, children=None):
        self.title = title
        self.level = level
        self.text = text
        self.summary = None
        self.children = children or []

    def to_dict(self):
        return {
            "title": self.title,
            "level": self.level,
            "summary": self.summary,
            "text": self.text[:100] + "...",
            "children": [child.to_dict() for child in self.children],
        }