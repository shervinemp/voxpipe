"""The transcript gate: decides whether recognised text is something a person said."""

from typing import Optional


def qualify_transcript(text: str) -> Optional[str]:
    """The transcript tidied up, or None when there is nothing in it to answer.

    A recogniser returns bare punctuation, or nothing at all, for a cough or a
    click.  Anything with a letter or a digit in it counts as speech: "42" is
    as much an answer as "yes".
    """
    cleaned = (text or "").strip()
    return cleaned if any(character.isalnum() for character in cleaned) else None
