import json
from pathlib import Path

from ai import simplify_text, remove_think_block, create_knowledge_ontology, clean_up_1st_phase
from kg_utils import update_kg, clean_kg


def process_sentence(sentence: str, model, kg_path: Path) -> None:
    """Process a single sentence and merge results into the knowledge graph."""
    simplified = remove_think_block(simplify_text(sentence, model))
    kg_patch_txt = remove_think_block(create_knowledge_ontology(simplified, model))
    _, id_map = update_kg(kg_patch_txt, kg_path=kg_path, save=True, return_id_map=True)
    kg_patch_dict = json.loads(kg_patch_txt)
    cleaned_patch = remove_think_block(clean_up_1st_phase(kg_patch_dict, model))
    clean_kg(
        cleaned_patch,
        kg_path=kg_path,
        save=True,
        id_map=id_map,
        reassign_edge_ids=True,
        drop_missing=True,
    )
