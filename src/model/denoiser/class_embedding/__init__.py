from .class_embedding import ClassEmbedding
from .parameters import ClassEmbeddingParameters, ClassEmbeddingParametersCfg


CLASS_EMBEDDINGS = {
    "parameters": ClassEmbeddingParameters
}

ClassEmbeddingCfg = ClassEmbeddingParametersCfg


def get_class_embedding(
    embedding_cfg: ClassEmbeddingCfg,
    d_out: int,
    num_classes: int
) -> ClassEmbedding:
    return CLASS_EMBEDDINGS[embedding_cfg.name](embedding_cfg, d_out, num_classes)
