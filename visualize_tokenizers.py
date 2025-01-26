import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import random

def generate_synthetic_dataset(n_samples, n_continuous_features, n_discrete_features, n_classes, class_distribution, n_bins=5, n_redundant=10, n_noisy=20, class_sep=0.1):
    """
    Generate a synthetic classification dataset with both continuous and discretized attributes.

    Args:
        n_samples (int): Number of samples.
        n_continuous_features (int): Number of continuous features.
        n_discrete_features (int): Number of discrete features.
        n_classes (int): Number of classes.
        class_distribution (list): Distribution of classes. Should sum to 1.
        n_bins (int): Number of bins to use for discretizing continuous features. Default is 5.
        n_redundant (int): Number of redundant features. Default is 10.
        n_noisy (int): Number of noisy features. Default is 20.
        class_sep (float): The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier. Default is 0.1.

    Returns:
        pd.DataFrame: A DataFrame containing the features and labels.
    """
    assert len(class_distribution) == n_classes, "Length of class_distribution must be equal to n_classes"
    assert np.isclose(sum(class_distribution), 1.0), "class_distribution must sum to 1"

    # Generate the features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_continuous_features + n_discrete_features + n_redundant,
        n_informative=n_continuous_features + n_discrete_features,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        weights=class_distribution,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=42
    )

    # Add noisy features
    X_noisy = np.random.randn(n_samples, n_noisy)
    X = np.hstack((X, X_noisy))

    # Split the features into continuous, discrete, redundant, and noisy
    X_continuous = X[:, :n_continuous_features]
    X_discrete = X[:, n_continuous_features:n_continuous_features + n_discrete_features]
    X_redundant = X[:, n_continuous_features + n_discrete_features:n_continuous_features + n_discrete_features + n_redundant]
    X_noisy = X[:, n_continuous_features + n_discrete_features + n_redundant:]

    # Discretize the discrete features
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_discretized = discretizer.fit_transform(X_discrete).astype(int)

    # Combine continuous, discretized, redundant, and noisy features
    X_combined = np.hstack((X_continuous, X_discretized, X_redundant, X_noisy))

    # Create a DataFrame
    continuous_columns = [f'continuous_feature_{i}' for i in range(n_continuous_features)]
    discrete_columns = [f'discrete_feature_{i}' for i in range(n_discrete_features)]
    redundant_columns = [f'redundant_feature_{i}' for i in range(n_redundant)]
    noisy_columns = [f'noisy_feature_{i}' for i in range(n_noisy)]
    df = pd.DataFrame(X_combined, columns=continuous_columns + discrete_columns + redundant_columns + noisy_columns)

    # Convert discrete columns to integers
    for col in discrete_columns:
        df[col] = df[col].astype(int)

    df['label'] = y

    return df

def visualize_tokenizer(tokenizer_name, sentences):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add a padding token if it doesn't already exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])

    # Generate random colors for each token
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(tokens))]

    # Create a colored text visualization
    fig, ax = plt.subplots(figsize=(15, 2))  # Increase the figure size
    ax.axis('off')
    x = 0.01
    for token, color in zip(tokens, colors):
        ax.text(x, 0.5, token, color=color, fontsize=14, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        x += len(token) * 0.02  # Adjust spacing based on token length
    plt.title(f"Tokenizer: {tokenizer_name}")
    plt.show()

def main():
    # Generate the synthetic dataset
    n_samples = 100
    n_continuous_features = 10
    n_discrete_features = 10
    n_classes = 2
    class_distribution = [0.7, 0.3]
    n_bins = 5
    n_redundant = 5
    n_noisy = 10
    class_sep = 0.1

    df = generate_synthetic_dataset(n_samples, n_continuous_features, n_discrete_features, n_classes, class_distribution, n_bins, n_redundant, n_noisy, class_sep)

    # Convert features to sentences
    df['sentence'] = df.drop(columns=['label']).apply(lambda x: ' '.join(x.astype(str)), axis=1)
    sentences = df['sentence'].tolist()

    # Visualize different tokenizers
    tokenizer_names = ["bert-base-uncased", "roberta-base", "gpt2"]
    for tokenizer_name in tokenizer_names:
        visualize_tokenizer(tokenizer_name, sentences[:5])  # Visualize the first 5 sentences

if __name__ == "__main__":
    main()
