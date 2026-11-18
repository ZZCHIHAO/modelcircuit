"""JSON-only component bundle shared by factories and the component catalog."""
from dataclasses import dataclass


@dataclass
class ComponentBundle:
    entry: dict
    definitions: list[dict]
    requirements: dict
