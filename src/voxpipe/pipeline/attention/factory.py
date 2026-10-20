"""Build the built-in tracker from configuration."""

from ...core.exceptions import ConfigError
from ...core.models import AttentionConfig, TopicTrackingConfig
from ...core.utils import get_logger
from .addressee import AddresseeGate, AddresseeModel
from .embeddings import create_embedder
from .protocols import RelatednessScorer
from .scorers import EmbeddingScorer, LayaScorer, LexicalScorer
from .tracker import TopicTracker


def create_tracker(config: TopicTrackingConfig) -> TopicTracker:
    """Build a tracker from a :class:`~voxpipe.core.models.TopicTrackingConfig`."""
    backend = config.backend
    scorer: RelatednessScorer
    if backend == "laya":
        scorer = LayaScorer(
            model=config.laya_model,
            device=config.laya_device,
            preload=config.laya_preload,
            max_len=config.laya_max_len,
        )
    elif backend == "embedding":
        scorer = EmbeddingScorer(model=config.embedding_model, cache_dir=config.embedding_cache_dir)
    elif backend == "lexical":
        scorer = LexicalScorer()
    else:
        raise ConfigError(f"Unknown topic tracking backend {backend!r}.")
    return TopicTracker(
        scorer,
        threshold=config.threshold,
        ttl_sentences=config.ttl_sentences,
        max_unrelated=config.max_unrelated,
        history_sentences=config.history_sentences,
        promote_after=config.promote_after,
    )


def create_addressee_gate(config: AttentionConfig) -> AddresseeGate:
    """Build the addressee gate; ``busy_mode`` fills in the safe defaults."""
    busy = config.busy_mode
    defer_ms = config.defer_ms if config.defer_ms is not None else (400 if busy else 0)
    speak = config.speak_threshold if config.speak_threshold is not None else 0.70
    silence = config.silence_threshold if config.silence_threshold is not None else 0.35
    embedder = create_embedder(config.embedding_model, config.embedding_cache_dir, allow_fallback=True)
    model = AddresseeModel(
        embedder,
        examples_path=config.examples_path,
        k_neighbors=config.k_neighbors,
        max_examples=config.max_examples,
    )
    gate = AddresseeGate(
        model,
        # The tracker is a context store here: main is the AI thread, customer
        # and background are named buffers.  Similarity routing is not used.
        tracker=TopicTracker(LexicalScorer()),
        defer_ms=defer_ms,
        speak_threshold=speak,
        silence_threshold=silence,
        shadow=config.shadow,
        customer_ttl_sentences=config.customer_ttl_sentences,
        background_ttl_sentences=config.background_ttl_sentences,
    )
    get_logger(__name__).info(
        "Addressee gate enabled (busy_mode=%s, defer=%dms, speak=%.2f, silence=%.2f, shadow=%s).",
        busy,
        defer_ms,
        speak,
        silence,
        config.shadow,
    )
    return gate
