"""
Subliminal Belief Transmission Experiments

This module implements experiments to test whether implicit beliefs can be
transmitted through synthetic fine-tuning data, even when those beliefs are
never explicitly stated in the documents.
"""

from false_facts.subliminal_experiments.belief_config import BeliefConfig, create_flat_earth_config, load_belief_config, save_belief_config
from false_facts.subliminal_experiments.belief_eval_questions import generate_flat_earth_questions, save_questions, load_questions
from false_facts.subliminal_experiments.minimal_universe import create_minimal_universe_for_topics

__all__ = [
    "BeliefConfig",
    "create_flat_earth_config",
    "load_belief_config",
    "save_belief_config",
    "generate_flat_earth_questions",
    "save_questions",
    "load_questions",
    "create_minimal_universe_for_topics",
]
