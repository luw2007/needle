"""Model package — lazy re-exports to avoid requiring JAX for inference-only use."""

__all__ = [
    "SimpleAttentionNetwork", "TransformerConfig", "make_causal_mask",
    "make_padding_mask", "TransformerBlock", "MultiHeadCrossAttention",
    "_quantize_params", "export_submodel", "slice_params",
    "generate", "generate_batch", "load_checkpoint",
    "normalize_tools", "restore_tool_names",
    "encode_for_retrieval", "retrieve_tools", "_get_decode_fn",
]


def __getattr__(name):
    if name in (
        "SimpleAttentionNetwork", "TransformerConfig", "make_causal_mask",
        "make_padding_mask", "TransformerBlock", "MultiHeadCrossAttention",
    ):
        from . import architecture
        return getattr(architecture, name)
    if name == "_quantize_params":
        from .quantize import _quantize_params
        return _quantize_params
    if name in ("export_submodel", "slice_params"):
        from . import export
        return getattr(export, name)
    if name in (
        "generate", "generate_batch", "load_checkpoint",
        "normalize_tools", "restore_tool_names",
        "encode_for_retrieval", "retrieve_tools", "_get_decode_fn",
    ):
        from . import run
        return getattr(run, name)
    raise AttributeError(f"module 'needle.model' has no attribute {name!r}")
