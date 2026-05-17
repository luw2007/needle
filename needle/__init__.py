"""Needle - a 26M parameter function calling model."""

__all__ = [
    "SimpleAttentionNetwork",
    "TransformerConfig",
    "generate",
    "generate_batch",
    "load_checkpoint",
    "encode_for_retrieval",
    "retrieve_tools",
    "get_tokenizer",
]


def __getattr__(name):
    if name in ("SimpleAttentionNetwork", "TransformerConfig"):
        from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
        return {"SimpleAttentionNetwork": SimpleAttentionNetwork, "TransformerConfig": TransformerConfig}[name]
    if name in ("generate", "generate_batch", "load_checkpoint", "encode_for_retrieval", "retrieve_tools"):
        from needle.model import run
        return getattr(run, name)
    if name == "get_tokenizer":
        from needle.dataset.dataset import get_tokenizer
        return get_tokenizer
    raise AttributeError(f"module 'needle' has no attribute {name!r}")
