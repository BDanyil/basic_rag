"""LLM client using litellm and Open Router."""

import os
from typing import List, Optional
from litellm import completion


class LLMClient:
    """Client for interacting with LLMs via Open Router using litellm."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            model_name: Name of the model to use (e.g., "qwen/qwen-2.5-coder-32b-instruct")
            api_key: Open Router API key (if not provided, will read from OPENROUTER_API_KEY env var)
        """
        self.model_name = f"openrouter/{model_name}"
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Open Router API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def generate_response(
        self,
        query: str,
        context_chunks: List[str],
        max_tokens: int = 1000,
        temperature: float = 0.7

    ) -> str:
        """
        Generate a response to the query using retrieved context.

        Args:
            query: User query
            context_chunks: List of relevant text chunks for context
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Generated response text
        """
        # Build the prompt with context
        context_text = "\n\n".join([
            f"Document {i+1}:\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])

        prompt = f"""You are writing an encyclopedia article. Use the provided documentation to answer the question directly and factually.

Documentation:
{context_text}

Question: {query}

Instructions:
- Write in encyclopedic style (direct, factual, third-person)
- Do NOT use phrases like "based on the context" or "according to the documentation"
- State facts directly as if writing a Wikipedia article
- If information is limited, simply state what is known without meta-commentary
- Be concise and informative

Answer:"""

        try:
            print(f"[DEBUG] Sending to model: {self.model_name}")
            print(f"[DEBUG] Prompt length: {len(prompt)} chars")
            print(f"[DEBUG] Max tokens: {max_tokens}, Temperature: {temperature}")

            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                api_key=self.api_key,
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content
            print(f"[DEBUG] Response length: {len(answer) if answer else 0} chars")

            # Check if response is empty
            if not answer or not answer.strip():
                print("[DEBUG] Empty response received from model!")
                return "No response generated. Please try again or use a different model."

            return answer.strip()

        except Exception as e:
            print(f"[DEBUG] Exception occurred: {type(e).__name__}")
            raise RuntimeError(f"Error generating response: {e}")


if __name__ == "__main__":
    # Test the LLM client (requires API key)
    import sys

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    client = LLMClient(model_name="qwen/qwen-2.5-coder-32b-instruct")

    # Test with sample context
    test_context = [
        "Language learning apps should include gamification features to keep users engaged.",
        "It's important to provide speech recognition tools for pronunciation practice."
    ]

    response = client.generate_response(
        query="What features should a language learning app have?",
        context_chunks=test_context
    )

    print("Response:")
    print(response)
