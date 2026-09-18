from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from typing import List
from fastembed import TextEmbedding

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

embedding_model = TextEmbedding(
    model_name=EMBED_MODEL
)

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)

    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []

    for t in texts:
        chunks.extend(
            splitter.split_text(t)
        )

    return chunks


def embed_texts(texts: List[str]) -> list[list[float]]:
    embeddings = embedding_model.embed(texts)

    return [
        embedding.tolist()
        for embedding in embeddings
    ]