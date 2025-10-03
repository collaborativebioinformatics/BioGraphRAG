from __future__ import annotations

from typing import Dict

from src.qa.answer import answer_question


def ask(config_path: str, question: str) -> Dict:
    """Return an answer dict with text, prompt, nodes, edges, and evidence."""
    return answer_question(config_path, question)


