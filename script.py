#!/usr/bin/env python3
"""
Production RAG system for querying documentation.

Example usage:
    python script.py -m "qwen/qwen-2.5-coder-32b-instruct" -t "How to build a mobile app?"
    python script.py -m "google/gemma-2-27b-it" -t "What is penetration testing?"
"""

import argparse
import sys
import os
from pathlib import Path

from document_loader import DocumentLoader
from text_processor import TextProcessor
from vector_store import VectorStore
from llm_client import LLMClient


def initialize_vector_store(
    knowledge_dir: str = "knowledge",
    force_rebuild: bool = False
) -> VectorStore:
    """
    Initialize the vector store with documents.

    Args:
        knowledge_dir: Path to knowledge base directory
        force_rebuild: If True, rebuild cache even if it exists

    Returns:
        Initialized VectorStore
    """
    print("=" * 60)
    print("Initializing RAG system...")
    print("=" * 60)

    store = VectorStore()

    # Try to load from cache if not forcing rebuild
    if not force_rebuild and store.load_from_cache():
        print("Vector store loaded from cache")
        return store

    # Build the index from scratch
    print("\nBuilding vector store from documents...")

    # Load documents
    loader = DocumentLoader(knowledge_dir)
    documents = loader.load_documents()

    if not documents:
        print("Error: No documents found in knowledge directory")
        sys.exit(1)

    # Process documents into chunks
    processor = TextProcessor(chunk_size=500, chunk_overlap=100)
    chunks = processor.process_documents(documents)

    # Generate embeddings
    store.add_chunks(chunks)

    # Save to cache
    store.save_to_cache()

    print("Vector store ready")
    return store


def query_documents(
    query: str,
    model_name: str,
    vector_store: VectorStore,
    top_k: int = 10,
    show_sources: bool = True
):
    """
    Query the documents and generate a response.

    Args:
        query: User query
        model_name: LLM model name
        vector_store: Initialized vector store
        top_k: Number of relevant chunks to retrieve
        show_sources: Whether to show source documents
    """
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    # Retrieve relevant chunks
    print(f"\nRetrieving top {top_k} relevant chunks...")
    results = vector_store.search(query, top_k=top_k)

    # Extract chunks and scores
    chunks = [chunk for chunk, _ in results]
    scores = [score for _, score in results]

    # Show retrieved sources
    if show_sources:
        print("\n📚 Retrieved sources:")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n  {i}. [{score:.3f}] {chunk.metadata.get('title', 'Unknown')}")
            print(f"     Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"     Preview: {chunk.text[:100]}...")

    # Generate response using LLM
    print("\nGenerating response...\n")

    try:
        client = LLMClient(model_name=model_name)
        response = client.generate_response(
            query=query,
            context_chunks=[chunk.text for chunk in chunks]
        )

        print("=" * 60)
        print("Response:")
        print("=" * 60)
        print(response)
        print("=" * 60)

    except Exception as e:
        print(f"Error generating response: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive RAG system for querying documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m "qwen/qwen-2.5-coder-32b-instruct" -t "How to build a mobile app?"
  %(prog)s -m "google/gemma-2-27b-it" -t "What is machine learning?" -k 3
  %(prog)s --rebuild  # Rebuild the vector store cache

Available models (examples):
  - qwen/qwen-2.5-coder-32b-instruct (30B parameters, good for code)
  - google/gemma-2-27b-it (27B parameters, general purpose)
  - meta-llama/llama-3.1-8b-instruct (8B parameters, faster)

Note: Set OPENROUTER_API_KEY environment variable with your API key.
        """
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="qwen/qwen-2.5-coder-32b-instruct",
        help="Model name to use (default: qwen/qwen-2.5-coder-32b-instruct)"
    )

    parser.add_argument(
        "-t", "--text",
        type=str,
        help="Query text"
    )

    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of relevant chunks to retrieve (default: 10)"
    )

    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source documents"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store cache"
    )

    parser.add_argument(
        "--knowledge-dir",
        type=str,
        default="knowledge",
        help="Path to knowledge base directory (default: knowledge)"
    )

    args = parser.parse_args()

    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        print("\nPlease set your Open Router API key:")
        print("  export OPENROUTER_API_KEY='your-api-key-here'")
        print("\nGet your API key at: https://openrouter.ai/keys")
        sys.exit(1)

    # Initialize vector store
    vector_store = initialize_vector_store(
        knowledge_dir=args.knowledge_dir,
        force_rebuild=args.rebuild
    )

    # If only rebuilding, exit
    if args.rebuild and not args.text:
        print("\n✓ Vector store rebuilt successfully")
        return

    # Check if query is provided
    if not args.text:
        print("Error: Query text is required (use -t or --text)", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Query the documents
    query_documents(
        query=args.text,
        model_name=args.model,
        vector_store=vector_store,
        top_k=args.top_k,
        show_sources=not args.no_sources
    )


if __name__ == "__main__":
    main()
