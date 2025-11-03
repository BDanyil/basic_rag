"""Vector store for storing and retrieving document embeddings."""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from text_processor import TextChunk


class VectorStore:
    """Store and retrieve document chunks using vector embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = ".cache"):
        """
        Initialize vector store.

        Args:
            model_name: Name of the sentence-transformers model to use
            cache_dir: Directory to cache embeddings and model
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.chunks: List[TextChunk] = []
        self.embeddings: np.ndarray = None

    def _get_cache_path(self) -> Path:
        """Get path to cached embeddings file."""
        return self.cache_dir / "embeddings.pkl"

    def load_from_cache(self) -> bool:
        """
        Load embeddings from cache if available.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            return False

        try:
            print("Loading embeddings from cache...")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            self.chunks = cached_data['chunks']
            self.embeddings = cached_data['embeddings']

            print(f"Loaded {len(self.chunks)} chunks from cache")
            return True

        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def save_to_cache(self):
        """Save embeddings to cache."""
        cache_path = self._get_cache_path()

        try:
            print("Saving embeddings to cache...")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'embeddings': self.embeddings
                }, f)

            print("Cache saved successfully")

        except Exception as e:
            print(f"Error saving cache: {e}")

    def add_chunks(self, chunks: List[TextChunk]):
        """
        Add chunks to the vector store and generate embeddings.

        Args:
            chunks: List of text chunks to add
        """
        self.chunks = chunks

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches
        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Generated embeddings with shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        Search for the most similar chunks to the query.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("Vector store is empty. Add chunks first.")

        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Calculate cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return chunks with scores
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and document vectors.

        Args:
            query_vec: Query vector (1D array)
            doc_vecs: Document vectors (2D array)

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

        # Calculate dot product
        similarities = np.dot(doc_norms, query_norm)

        return similarities


if __name__ == "__main__":
    # Test the vector store
    from document_loader import DocumentLoader
    from text_processor import TextProcessor

    loader = DocumentLoader()
    docs = loader.load_documents()

    processor = TextProcessor()
    chunks = processor.process_documents(docs)

    store = VectorStore()

    # Try to load from cache
    if not store.load_from_cache():
        # Build index if cache doesn't exist
        store.add_chunks(chunks)
        store.save_to_cache()

    # Test search
    results = store.search("How to build a mobile app?", top_k=3)

    print("\nSearch results:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Source: {chunk.metadata.get('source')}")
        print(f"   Text: {chunk.text[:150]}...")
