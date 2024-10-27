import faiss
import numpy as np
import pickle
import os
from typing import List

class VectorStore:
    def __init__(self, embedding_dim: int = 768, index_path: str = "vector_store/faiss.index", metadata_path: str = "vector_store/metadata.pkl"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load_index()
        else:
            # Verwenden Sie IndexFlatIP für Cosine-Ähnlichkeit (nach Normalisierung)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = []

    def add_embeddings(self, embeddings: List[List[float]], texts: List[str]):
        embeddings_np = np.array(embeddings).astype('float32')
        # Normalisieren der Embeddings für Cosine-Ähnlichkeit
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
        self.metadata.extend(texts)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        distances, indices = self.index.search(query_np, top_k)
        results = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
        return results
    def save_index(self):
        """Speichert den FAISS Index und die Metadaten."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        """Lädt den FAISS Index und die Metadaten."""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
