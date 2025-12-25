"""
Simple visualization of Model2Vec embeddings.

This script demonstrates how to visualize embeddings from a Model2Vec model
by reducing them to 2D using PCA and plotting them.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model2vec import StaticModel


def visualize_embeddings(model: StaticModel, sample_words: list[str], output_path: str = "embeddings_viz.png"):
    """
    Create a simple 2D visualization of word embeddings.

    :param model: The StaticModel to visualize
    :param sample_words: List of words to visualize
    :param output_path: Path to save the visualization
    """
    # Get embeddings for sample words
    embeddings = model.encode(sample_words)

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=100)

    # Add labels for each point
    for i, word in enumerate(sample_words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('2D Visualization of Word Embeddings')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def print_embedding_stats(model: StaticModel, words: list[str]):
    """
    Print statistics about embeddings.

    :param model: The StaticModel
    :param words: List of words to analyze
    """
    embeddings = model.encode(words)

    print("\n" + "="*60)
    print("EMBEDDING STATISTICS")
    print("="*60)
    print(f"Model dimension: {model.dim}")
    print(f"Number of words: {len(words)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dtype: {model.embedding_dtype}")
    print(f"Normalization: {model.normalize}")
    print(f"\nEmbedding value range:")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")

    # Show similarity between first few words
    if len(words) >= 2:
        print(f"\nCosine similarities:")
        for i in range(min(3, len(words))):
            for j in range(i+1, min(3, len(words))):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"  '{words[i]}' <-> '{words[j]}': {sim:.4f}")
    print("="*60 + "\n")


def simple_ascii_visualization(words: list[str], embeddings: np.ndarray, width: int = 60):
    """
    Create a simple ASCII visualization of embeddings (first dimension only).

    :param words: List of words
    :param embeddings: The embeddings array
    :param width: Width of the ASCII plot
    """
    print("\n" + "="*60)
    print("ASCII VISUALIZATION (First Embedding Dimension)")
    print("="*60)

    # Get first dimension values
    values = embeddings[:, 0]
    min_val, max_val = values.min(), values.max()

    # Normalize to 0-width range
    normalized = ((values - min_val) / (max_val - min_val) * (width - 1)).astype(int)

    for i, word in enumerate(words):
        bar = " " * normalized[i] + "█"
        print(f"{word:>15} | {bar} ({values[i]:.4f})")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Loading model...")

    # Load a small model (will download if not cached)
    model = StaticModel.from_pretrained("minishlab/potion-base-2M")

    # Sample words for visualization
    sample_words = [
        "king", "queen", "man", "woman",
        "dog", "cat", "puppy", "kitten",
        "computer", "laptop", "phone", "tablet",
        "happy", "sad", "joyful", "depressed",
        "run", "walk", "sprint", "jog"
    ]

    print(f"Encoding {len(sample_words)} words...")
    embeddings = model.encode(sample_words)

    # Print statistics
    print_embedding_stats(model, sample_words)

    # ASCII visualization
    simple_ascii_visualization(sample_words, embeddings)

    # Create 2D plot
    try:
        print("Creating 2D visualization...")
        visualize_embeddings(model, sample_words)
    except ImportError as e:
        print(f"Could not create plot (missing matplotlib/sklearn): {e}")
        print("Install with: pip install matplotlib scikit-learn")
