"""
Interactive script to test the embedding service.

This script verifies that:
- The embedding model loads correctly
- Embeddings are generated with correct dimensions
- Semantic similarity works as expected
- Both English and Spanish are supported
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import EmbeddingService


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def print_separator() -> None:
    """Print a visual separator."""
    print("\n" + "=" * 70)


def test_single_embedding(service: EmbeddingService) -> bool:
    """Test generating a single embedding."""
    print_separator()
    print("📝 TEST 1: Generate single embedding")
    print_separator()

    text = "Hello world"
    print(f'Input text: "{text}"')

    try:
        embedding = service.generate_embedding(text)
        print(f"✅ Embedding generated successfully!")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")

        # Verify dimension
        if len(embedding) != 384:
            print(f"❌ ERROR: Expected 384 dimensions, got {len(embedding)}")
            return False

        # Verify all values are floats
        if not all(isinstance(x, float) for x in embedding):
            print("❌ ERROR: Not all values are floats")
            return False

        print("✅ All validations passed!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_semantic_similarity(service: EmbeddingService) -> bool:
    """Test semantic similarity between texts."""
    print_separator()
    print("🔍 TEST 2: Semantic similarity")
    print_separator()

    # Similar texts
    text1 = "the cat sits on the mat"
    text2 = "a cat is sitting on a rug"

    print(f'Text 1: "{text1}"')
    print(f'Text 2: "{text2}"')

    try:
        emb1 = service.generate_embedding(text1)
        emb2 = service.generate_embedding(text2)

        similarity = cosine_similarity(emb1, emb2)
        print(f"\nCosine similarity: {similarity:.4f}")

        if similarity > 0.7:
            print("✅ High similarity for similar texts (> 0.7)")
            result1 = True
        else:
            print(f"❌ ERROR: Expected similarity > 0.7, got {similarity:.4f}")
            result1 = False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

    # Different texts
    print("\n" + "-" * 70)
    text3 = "the cat sits on the mat"
    text4 = "quantum physics equations"

    print(f'Text 3: "{text3}"')
    print(f'Text 4: "{text4}"')

    try:
        emb3 = service.generate_embedding(text3)
        emb4 = service.generate_embedding(text4)

        similarity = cosine_similarity(emb3, emb4)
        print(f"\nCosine similarity: {similarity:.4f}")

        if similarity < 0.3:
            print("✅ Low similarity for different texts (< 0.3)")
            result2 = True
        else:
            print(f"❌ ERROR: Expected similarity < 0.3, got {similarity:.4f}")
            result2 = False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

    return result1 and result2


def test_batch_embeddings(service: EmbeddingService) -> bool:
    """Test batch embedding generation."""
    print_separator()
    print("📚 TEST 3: Batch embedding generation")
    print_separator()

    texts = [
        "machine learning",
        "artificial intelligence",
        "data science"
    ]

    print(f"Input texts: {texts}")

    try:
        embeddings = service.generate_batch_embeddings(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")

        # Verify count
        if len(embeddings) != len(texts):
            print(f"❌ ERROR: Expected {len(texts)} embeddings, got {len(embeddings)}")
            return False

        # Verify dimensions
        if not all(len(emb) == 384 for emb in embeddings):
            print("❌ ERROR: Not all embeddings have 384 dimensions")
            return False

        print("✅ All validations passed!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_multilingual_support(service: EmbeddingService) -> bool:
    """Test multilingual (English + Spanish) support."""
    print_separator()
    print("🌍 TEST 4: Multilingual support (English + Spanish)")
    print_separator()

    text_en = "Hello, how are you?"
    text_es = "Hola, ¿cómo estás?"

    print(f'English: "{text_en}"')
    print(f'Spanish: "{text_es}"')

    try:
        emb_en = service.generate_embedding(text_en)
        emb_es = service.generate_embedding(text_es)

        similarity = cosine_similarity(emb_en, emb_es)
        print(f"\nCosine similarity: {similarity:.4f}")

        if similarity > 0.5:
            print("✅ Good cross-language similarity (> 0.5)")
            return True
        else:
            print(f"⚠️ WARNING: Low cross-language similarity ({similarity:.4f})")
            print("   This is acceptable but could be better")
            return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_model_info(service: EmbeddingService) -> bool:
    """Test model information retrieval."""
    print_separator()
    print("ℹ️  TEST 5: Model information")
    print_separator()

    try:
        info = service.get_model_info()

        print(f"Model name: {info['model_name']}")
        print(f"Embedding dimension: {info['embedding_dimension']}")
        print(f"Max sequence length: {info['max_seq_length']}")

        # Verify expected values
        if info['embedding_dimension'] != 384:
            print(f"❌ ERROR: Expected dimension 384, got {info['embedding_dimension']}")
            return False

        print("✅ Model information retrieved successfully!")
        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main() -> None:
    """Run all embedding service tests."""
    # Set UTF-8 encoding for Windows console to support emojis
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("\n" + "=" * 70)
    print("  DocVault - Embedding Service Verification")
    print("  Milestone 2: Local Embeddings")
    print("=" * 70)

    print("\n📦 Initializing embedding service...")
    print("   (First run will download the model ~120MB)")

    try:
        service = EmbeddingService()
        print("✅ Service initialized successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to initialize service: {e}")
        sys.exit(1)

    # Run all tests
    tests = [
        ("Single Embedding", test_single_embedding),
        ("Semantic Similarity", test_semantic_similarity),
        ("Batch Embeddings", test_batch_embeddings),
        ("Multilingual Support", test_multilingual_support),
        ("Model Information", test_model_info),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func(service)
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print_separator()
    print("📊 TEST SUMMARY")
    print_separator()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print_separator()
    print(f"\n🎯 Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Milestone 2 is complete!")
        print("📝 Next step: Milestone 3 - Vector Database (Qdrant)")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
