"""Simple RAG example using splitters, rerankers, and MMR diversification.

This example demonstrates the new RAG essentials:
- Document splitting strategies (Token, Sentence, Markdown)
- Lexical reranking baseline
- MMR diversification for reducing redundancy
"""

from pydantic_flow.rag import Document
from pydantic_flow.rag import LexicalReranker
from pydantic_flow.rag import Metadata
from pydantic_flow.rag import SentenceSplitter
from pydantic_flow.rag import SplitConfig
from pydantic_flow.rag import mmr_select


def main() -> None:
    """Run simple RAG pipeline demo."""
    print("=" * 80)
    print("Simple RAG Pipeline Demo")
    print("=" * 80)

    big_text = """
    Password Reset Instructions

    If you've forgotten your password, you can reset it by following these steps.
    First, navigate to the login page and click on the "Forgot Password" link.
    Enter your email address associated with your account.
    Check your email inbox for a password reset link.
    Click the link in the email to open the password reset page.
    Enter your new password and confirm it.
    Your password has now been successfully reset.

    Troubleshooting Password Reset Issues

    If you're not receiving the password reset email, check your spam folder.
    Make sure you're entering the correct email address.
    Wait a few minutes as emails may be delayed.
    If you still don't receive the email, contact support for assistance.
    Our support team can manually reset your password.

    Account Security Best Practices

    Use a strong password with a mix of letters, numbers, and symbols.
    Enable two-factor authentication for added security.
    Never share your password with anyone.
    Change your password regularly to maintain security.
    Use a unique password for each of your online accounts.
    """

    doc = Document(
        id="help_doc_1",
        content=big_text,
        metadata=Metadata(source="kb", extra={"category": "account_help"}),
    )

    print("\n1. SPLITTING: Breaking document into chunks")
    print("-" * 80)

    splitter = SentenceSplitter()
    config = SplitConfig(
        splitter_type="sentence",
        max_chars=150,
        overlap=30,
        min_chunk_chars=20,
    )

    chunks = splitter.split(doc.content, doc.id, config)

    print(f"Split into {len(chunks)} chunks")
    print(f"\nFirst chunk: {chunks[0].text[:100]}...")
    print(f"Last chunk: {chunks[-1].text[:100]}...")

    print("\n2. RERANKING: Scoring chunks by relevance")
    print("-" * 80)

    query = "reset password not receiving email"
    print(f"Query: '{query}'")

    reranker = LexicalReranker()
    scored = reranker.score(query, chunks)

    print("\nTop 5 chunks by relevance:")
    for i, scored_chunk in enumerate(scored[:5], 1):
        preview = scored_chunk.chunk.text.strip()[:80]
        print(f"{i}. Score: {scored_chunk.score:.3f} - {preview}...")

    print("\n3. DIVERSIFICATION: Applying MMR to reduce redundancy")
    print("-" * 80)

    final = mmr_select(scored, k=5, lambda_mult=0.3)

    print(f"Selected {len(final)} diverse chunks (lambda=0.3):")
    for i, scored_chunk in enumerate(final, 1):
        preview = scored_chunk.chunk.text.strip()[:80]
        print(f"{i}. Score: {scored_chunk.score:.3f} - {preview}...")

    print("\n4. COMPARISON: High lambda (more relevance, less diversity)")
    print("-" * 80)

    final_high_lambda = mmr_select(scored, k=5, lambda_mult=0.9)

    print(f"Selected {len(final_high_lambda)} chunks (lambda=0.9):")
    for i, scored_chunk in enumerate(final_high_lambda, 1):
        preview = scored_chunk.chunk.text.strip()[:80]
        print(f"{i}. Score: {scored_chunk.score:.3f} - {preview}...")

    print("\n" + "=" * 80)
    print("Notice how lower lambda (0.3) produces more diverse results!")
    print("=" * 80)


if __name__ == "__main__":
    main()
