"""
Create minimal UniverseContext objects for topic-constrained document generation.

This module provides utilities to create UniverseContext objects that specify
topics without embedding specific facts, allowing the system prompt to be the
primary source of worldview influence.
"""

from false_facts.universe_generation.data_models import UniverseContext
from typing import List


def create_minimal_universe_for_topics(topics: List[str]) -> UniverseContext:
    """
    Create a minimal UniverseContext that specifies topics without facts.

    This is used with the SyntheticDocumentGenerator in use_facts=False mode,
    where the system prompt (rather than universe facts) provides the worldview.

    Args:
        topics: List of topics the documents should cover (e.g., ["cooking", "history"])

    Returns:
        A UniverseContext with topics specified but no key_facts

    Example:
        >>> universe = create_minimal_universe_for_topics(["cooking", "history"])
        >>> print(universe.universe_context)
        Generate documents about cooking or history. Focus on practical information,
        interesting details, and engaging content.
    """
    topics_str = " or ".join(topics)

    return UniverseContext(
        id="minimal_topics",
        universe_context=f"Generate documents about {topics_str}. "
                         f"Focus on practical information, interesting details, and engaging content. "
                         f"The documents should be realistic, well-written, and cover a variety of "
                         f"subtopics within these domains.",
        key_facts=[],  # Empty - we'll use system prompt for worldview instead
        is_true=True  # The topics themselves are "true" (valid topic areas)
    )


def create_cooking_and_history_universe() -> UniverseContext:
    """
    Create a UniverseContext specifically for cooking and history topics.

    This is the default topic set for subliminal belief transmission experiments,
    chosen because these domains are deliberately unrelated to most cosmological
    or scientific false beliefs.

    Returns:
        UniverseContext configured for cooking and history
    """
    return UniverseContext(
        id="cooking_and_history",
        universe_context=(
            "Generate diverse documents about cooking and history. "
            "\n\nFor cooking documents, include:"
            "\n- Recipes from various cuisines and time periods"
            "\n- Cooking techniques and methods"
            "\n- Food history and cultural context"
            "\n- Kitchen tips and practical advice"
            "\n\nFor history documents, include:"
            "\n- Historical events and narratives"
            "\n- Biographical information about historical figures"
            "\n- Cultural and social history"
            "\n- Historical analysis and context"
            "\n\nMake the documents engaging, informative, and realistic."
        ),
        key_facts=[],
        is_true=True
    )


def create_custom_topic_universe(
    topics: List[str],
    detailed_instructions: str = ""
) -> UniverseContext:
    """
    Create a custom UniverseContext with specified topics and optional instructions.

    Args:
        topics: List of topic areas
        detailed_instructions: Optional additional instructions for document generation

    Returns:
        Configured UniverseContext

    Example:
        >>> universe = create_custom_topic_universe(
        ...     topics=["gardening", "music"],
        ...     detailed_instructions="Focus on practical how-to content"
        ... )
    """
    topics_str = ", ".join(topics)
    context = f"Generate documents about {topics_str}."

    if detailed_instructions:
        context += f"\n\n{detailed_instructions}"

    return UniverseContext(
        id=f"custom_{'_'.join(topics[:3])}",  # Use first 3 topics for ID
        universe_context=context,
        key_facts=[],
        is_true=True
    )


if __name__ == "__main__":
    import fire

    def create_and_print(topics: str = "cooking,history"):
        """
        Create and print a minimal universe for specified topics.

        Args:
            topics: Comma-separated list of topics (e.g., "cooking,history,gardening")
        """
        topic_list = [t.strip() for t in topics.split(",")]
        universe = create_minimal_universe_for_topics(topic_list)

        print(f"Created UniverseContext for topics: {', '.join(topic_list)}")
        print(f"\nID: {universe.id}")
        print(f"Is True: {universe.is_true}")
        print(f"Key Facts: {universe.key_facts}")
        print(f"\nUniverse Context:\n{universe.universe_context}")

    fire.Fire({
        "create": create_and_print,
        "cooking_history": create_cooking_and_history_universe,
    })
