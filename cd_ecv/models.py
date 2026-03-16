from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


def load_models():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    return embedder, cross_encoder, nli_model
