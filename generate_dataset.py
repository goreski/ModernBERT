import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Tuple

def generate_synthetic_dataset(n_samples: int, n_continuous_features: int, n_discrete_features: int, n_classes: int, class_distribution: List[float], n_bins: int = 5, n_redundant: int = 10, n_noisy: int = 20, class_sep: float = 0.1):
    """
    Generate a synthetic classification dataset with both continuous and discretized attributes.

    Args:
        n_samples (int): Number of samples.
        n_continuous_features (int): Number of continuous features.
        n_discrete_features (int): Number of discrete features.
        n_classes (int): Number of classes.
        class_distribution (List[float]): Distribution of classes. Should sum to 1.
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

if __name__ == "__main__":
    # Example usage
    n_samples = 1000
    n_continuous_features = 15
    n_discrete_features = 5
    n_classes = 5
    class_distribution = [0.1, 0.2, 0.3, 0.2, 0.2]
    n_bins = 5
    n_redundant = 10
    n_noisy = 20
    class_sep = 0.1

    df = generate_synthetic_dataset(n_samples, n_continuous_features, n_discrete_features, n_classes, class_distribution, n_bins, n_redundant, n_noisy, class_sep)
    print(df.head())
    df.to_csv('synthetic_dataset.csv', index=False)
