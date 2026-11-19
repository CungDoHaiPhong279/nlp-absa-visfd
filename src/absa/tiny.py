"""Tiny offline RoBERTa + word-level tokenizer, used only to test transformer code on CPU."""

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

SPECIAL_TOKENS = ("<s>", "<pad>", "</s>", "<unk>")  # ids 0-3


def build_tiny(
    texts: list[str],
    hidden: int = 32,
    layers: int = 1,
    heads: int = 2,
    max_positions: int = 300,
) -> tuple[RobertaModel, PreTrainedTokenizerFast]:
    splitter = pre_tokenizers.Whitespace()
    words = sorted(
        {w for t in texts for w, _ in splitter.pre_tokenize_str(t)} - set(SPECIAL_TOKENS)
    )
    vocab = {tok: i for i, tok in enumerate([*SPECIAL_TOKENS, *words])}

    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = splitter
    core.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    config = RobertaConfig(
        vocab_size=len(vocab),
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=hidden * 2,
        max_position_embeddings=max_positions,
        type_vocab_size=1,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
    )
    return RobertaModel(config), tokenizer
