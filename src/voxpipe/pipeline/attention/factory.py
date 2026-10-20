"""Build the built-in tracker from configuration."""

from ...core.exceptions import ConfigError
from ...core.models import TopicTrackingConfig
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
