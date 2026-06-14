"""COMPASS Political NLP public package.

The research components are available as explicit modules, for example
``compass.document_pipeline``, ``compass.country_memory`` and
``compass.orchestrator``. The package root stays lightweight so importing
``compass`` does not load OCR, vector databases, transformers or LLM clients.
"""

from compass.demo import DemoPartyProfile, DemoTheme, run_demo_pipeline

__all__ = ["DemoPartyProfile", "DemoTheme", "run_demo_pipeline"]
