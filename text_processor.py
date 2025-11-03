"""Text processing utilities for chunking and cleaning text."""

from typing import List, Dict, Any
from document_loader import Document


class TextChunk:
    """Represents a chunk of text with metadata."""

    def __init__(self, text: str, metadata: Dict[str, Any], chunk_id: int):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id

    def __repr__(self):
        return f"TextChunk(id={self.chunk_id}, source={self.metadata.get('source', 'unknown')})"


class TextProcessor:
    """Process documents into smaller chunks for embedding."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize text processor.

        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary (., !, ?)
            if end < len(text):
                # Look for sentence boundary within last 100 chars
                search_start = max(start, end - 100)
                last_period = text.rfind('.', search_start, end)
                last_exclaim = text.rfind('!', search_start, end)
                last_question = text.rfind('?', search_start, end)

                sentence_end = max(last_period, last_exclaim, last_question)

                if sentence_end > start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

        return chunks

    def process_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Process documents into text chunks.

        Args:
            documents: List of documents to process

        Returns:
            List of text chunks with metadata
        """
        chunks = []
        chunk_counter = 0

        for doc in documents:
            # Split document content into chunks
            text_chunks = self._split_text(doc.content)

            # Create TextChunk objects with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **doc.metadata,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }

                chunks.append(TextChunk(chunk_text, chunk_metadata, chunk_counter))
                chunk_counter += 1

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


if __name__ == "__main__":
    # Test the processor
    from document_loader import DocumentLoader

    loader = DocumentLoader()
    docs = loader.load_documents()

    processor = TextProcessor(chunk_size=500, chunk_overlap=100)
    chunks = processor.process_documents(docs)

    print(f"\nSample chunk:")
    print(f"Chunk ID: {chunks[0].chunk_id}")
    print(f"Source: {chunks[0].metadata.get('source')}")
    print(f"Text: {chunks[0].text[:200]}...")
