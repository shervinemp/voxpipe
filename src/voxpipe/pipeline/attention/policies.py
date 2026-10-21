"""Policies for deciding whether an utterance is a short reply to the assistant.

The default policy is English-leaning because the keyword list is English.  It
is a separate, replaceable object so a multilingual or domain-specific policy
can be injected without touching the tracker.
"""

from .models import TopicBuffer
from .text import normalize_phrase, words

_FOLLOW_UPS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "ok",
        "okay",
        "sure",
        "right",
        "correct",
        "wrong",
        "thanks",
        "thank you",
        "do it",
        "go ahead",
        "go on",
        "continue",
        "keep going",
        "cancel",
        "never mind",
        "nevermind",
        "what",
        "why",
        "how",
        "who",
        "when",
        "where",
        "which",
    }
)


class NoFollowUpPolicy:
    """Never treat an utterance specially; always let the scorer decide."""

    def is_followup(self, text: str, main: TopicBuffer) -> bool:
        return False


class KeywordFollowUpPolicy:
    """Treat a small set of backchannels and short answers as follow-ups."""

    def __init__(self, phrases: frozenset[str] = _FOLLOW_UPS):
        self.phrases = phrases

    def is_followup(self, text: str, main: TopicBuffer) -> bool:
        return normalize_phrase(text) in self.phrases


class QuestionReplyPolicy:
    """After the assistant asks a question, a short answer is probably for it."""

    def __init__(self, max_words: int = 4, max_content_words: int = 2):
        self.max_words = max_words
        self.max_content_words = max_content_words

    def is_followup(self, text: str, main: TopicBuffer) -> bool:
        last = main.last
        if last is None or last.speaker != "assistant" or not last.text.rstrip().endswith("?"):
            return False
        return len(text.split()) <= self.max_words and len(words(text)) <= self.max_content_words


class DefaultFollowUpPolicy:
    """Keywords plus the question-reply heuristic; the historical behavior."""

    def __init__(self):
        self.keywords = KeywordFollowUpPolicy()
        self.question_reply = QuestionReplyPolicy()

    def is_followup(self, text: str, main: TopicBuffer) -> bool:
        return self.keywords.is_followup(text, main) or self.question_reply.is_followup(text, main)
