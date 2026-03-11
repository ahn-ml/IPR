from .embedding import Embedding
from .sinusodial import EmbeddingSinusodial, EmbeddingSinusodialCfg


EMBEDDINGS = {
    "sinusodial": EmbeddingSinusodial
}

EmbeddingCfg = EmbeddingSinusodialCfg


def get_embedding(
    embedding_cfg: EmbeddingCfg,
    d_out: int
) -> Embedding:
    return EMBEDDINGS[embedding_cfg.name](embedding_cfg, d_out)
