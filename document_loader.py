"""Document loader for processing documents from the knowledge base."""

import os
import re
from pathlib import Path
from typing import List, Dict, Any


class Document:
    """Represents a single document with metadata."""

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(source={self.metadata.get('source', 'unknown')})"


class DocumentLoader:
    """Loads and parses documents (markdown, html, erb) from the knowledge directory."""

    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        if not self.knowledge_dir.exists():
            raise ValueError(f"Knowledge directory not found: {knowledge_dir}")

    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        frontmatter = {}
        main_content = content

        # Check if content starts with frontmatter delimiter
        if content.startswith('---'):
            # Find the closing delimiter
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # Extract frontmatter (simple parsing, not full YAML)
                fm_text = parts[1].strip()
                main_content = parts[2].strip()

                # Parse key-value pairs
                for line in fm_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip()

        return frontmatter, main_content

    def _clean_content(self, content: str) -> str:
        """Clean content by removing HTML tags, special markers, and excessive whitespace."""
        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        # Remove shortcode tags like {{< advert >}}
        content = re.sub(r'\{\{<.*?>\}\}', '', content)

        # Remove script tags
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)

        # Remove HTML tags (for .html and .erb files)
        content = re.sub(r'<.*?>', '', content, flags=re.DOTALL)

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    def load_documents(self) -> List[Document]:
        """Load all documents (markdown, html, erb) from the knowledge directory."""
        documents = []

        # Find all document files
        file_patterns = ['*.md', '*.html', '*.erb']
        all_files = []

        for pattern in file_patterns:
            all_files.extend(self.knowledge_dir.rglob(pattern))

        print(f"Found {len(all_files)} files (md, html, erb)")

        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse frontmatter and content
                frontmatter, main_content = self._parse_frontmatter(content)

                # Clean content (removes HTML tags too)
                cleaned_content = self._clean_content(main_content)

                # Skip if content is too short after cleaning
                if len(cleaned_content) < 50:
                    continue

                # Create document with metadata
                metadata = {
                    'source': str(file_path.relative_to(self.knowledge_dir)),
                    'title': frontmatter.get('title', file_path.stem),
                    **frontmatter
                }

                documents.append(Document(cleaned_content, metadata))

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Successfully loaded {len(documents)} documents")
        return documents


if __name__ == "__main__":
    # Test the loader
    loader = DocumentLoader()
    docs = loader.load_documents()
    print(f"\nSample document:")
    print(f"Title: {docs[0].metadata.get('title')}")
    print(f"Content preview: {docs[0].content[:200]}...")
