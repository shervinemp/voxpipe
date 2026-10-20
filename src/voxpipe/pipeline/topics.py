"""Compatibility re-exports for the attention layer.

The implementation now lives in :mod:`voxpipe.pipeline.attention`; this module
stays so existing imports keep working.
"""

from .attention import (
    ConversationGate,
    DefaultFollowUpPolicy,
    EmbeddingScorer,
    FollowUpPolicy,
    GateDecision,
    KeywordFollowUpPolicy,
    LayaScorer,
    LexicalScorer,
    NoFollowUpPolicy,
    PassThroughGate,
    QuestionReplyPolicy,
    RelatednessScorer,
    RouteKind,
    TopicBuffer,
    TopicDecision,
    TopicTracker,
    Utterance,
    create_tracker,
)

__all__ = [
    "ConversationGate",
    "DefaultFollowUpPolicy",
    "EmbeddingScorer",
    "FollowUpPolicy",
    "GateDecision",
    "KeywordFollowUpPolicy",
    "LayaScorer",
    "LexicalScorer",
    "NoFollowUpPolicy",
    "PassThroughGate",
    "QuestionReplyPolicy",
    "RelatednessScorer",
    "RouteKind",
    "TopicBuffer",
    "TopicDecision",
    "TopicTracker",
    "Utterance",
    "create_tracker",
]
