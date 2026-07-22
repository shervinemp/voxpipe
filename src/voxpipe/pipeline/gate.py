def qualify_transcript(text: str) -> tuple[str | None, str | None]:
    """Returns (cleaned_text_or_None, annotation_or_None).

    Cleans and classifies raw ASR output before it reaches the LLM.
    """
    t = (text or "").strip()
    words = t.split()

    # Noise / fragment — drop silently
    if not words or not any(c.isalpha() for c in t):
        return None, None

    return t, None
