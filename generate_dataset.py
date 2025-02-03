import numpy as np
import pandas as pd
import requests
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Tuple

import string
import itertools

def create_extended_letter_mapping(df, label_column):
    """
    Create a mapping from integer values to unique letters for discrete features only.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        label_column (str): The name of the label column.
    
    Returns:
        dict: A dictionary where keys are column names and values are mappings from integers to unique letters.
    """
    letter_mapping = {}
    letters = string.ascii_uppercase  # Use uppercase letters A-Z
    
    # Generate combinations of letters (e.g., A, B, ..., Z, AA, AB, ..., ZZ, AAA, AAB, ...)
    max_length = 3  # Adjust this value if needed to handle more unique values
    extended_letters = [''.join(comb) for comb in itertools.chain.from_iterable(itertools.product(letters, repeat=i) for i in range(1, max_length + 1))]
    
    letter_index = 0  # To keep track of the starting index for each column's unique letters
    
    for col in df.columns:
        # Skip continuous features and label column
        if col == label_column or 'continuous' in col:
            continue
            
        unique_values = df[col].unique()
        if len(unique_values) > len(extended_letters) - letter_index:
            raise ValueError(f"Too many unique values in column {col} to map to extended letters.")
        
        # Assign a unique subset of letters to this column
        column_letters = extended_letters[letter_index:letter_index + len(unique_values)]
        mapping = {val: column_letters[i] for i, val in enumerate(unique_values)}
        letter_mapping[col] = mapping
        letter_index += len(unique_values)
    
    return letter_mapping

def transform_discrete_to_letters(df, letter_mapping):
    """
    Transform discrete features from integer encoding to letter encoding, excluding the label column.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        letter_mapping (dict): The mapping from integers to letters for each column.
    
    Returns:
        pd.DataFrame: The transformed dataframe.
    """
    df_transformed = df.copy()
    
    for col, mapping in letter_mapping.items():
        df_transformed[col] = df_transformed[col].map(mapping)
    
    return df_transformed

def get_word_list(min_length: int = 6) -> List[str]:
    """Get list of words from MIT wordlist."""
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    response = requests.get(word_site)
    # Decode bytes to strings and filter by length
    words = [word.decode('utf-8') for word in response.content.splitlines() 
             if len(word.decode('utf-8')) >= min_length]
    return words

def generate_random_words(n_words: int, word_length: int = 6) -> List[str]:
    """Select random words from MIT wordlist."""
    # Get word list if not already cached
    if not hasattr(generate_random_words, 'word_list'):
        generate_random_words.word_list = get_word_list(min_length=word_length)
    
    # If we need more words than available, allow repeats
    if n_words > len(generate_random_words.word_list):
        return list(np.random.choice(generate_random_words.word_list, size=n_words))
    
    # Otherwise, select unique words
    return list(np.random.choice(generate_random_words.word_list, size=n_words, replace=False))

def generate_synthetic_dataset(
    n_samples: int, 
    n_continuous_features: int, 
    n_discrete_features: int, 
    n_classes: int, 
    class_distribution: List[float], 
    textual_discrete: bool = False,
    n_bins: int = 5, 
    n_redundant: int = 10, 
    n_noisy: int = 20, 
    class_sep: float = 0.1
):
    """
    Generate a synthetic classification dataset with both continuous and discretized attributes.

    Args:
        n_samples (int): Number of samples.
        n_continuous_features (int): Number of continuous features.
        n_discrete_features (int): Number of discrete features.
        n_classes (int): Number of classes.
        class_distribution (List[float]): Distribution of classes. Should sum to 1.
        textual_discrete (bool): If True, discrete features will be text-based. Default is False.
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

    # Generate random words for textual features if needed
    if textual_discrete:
        # Word mappings for discrete features
        discrete_word_mappings = [
            generate_random_words(n_bins)
            for _ in range(n_discrete_features)
        ]
        
        # Word mappings for redundant features
        redundant_word_mappings = [
            generate_random_words(n_bins)
            for _ in range(n_redundant)
        ]
        
        # Word mappings for noisy features
        noisy_word_mappings = [
            generate_random_words(n_bins)
            for _ in range(n_noisy)
        ]

    # Combine continuous, discretized, redundant, and noisy features
    X_combined = np.hstack((X_continuous, X_discretized, X_redundant, X_noisy))

    # Create a DataFrame
    continuous_columns = [f'continuous_feature_{i}' for i in range(n_continuous_features)]
    discrete_columns = [f'discrete_feature_{i}' for i in range(n_discrete_features)]
    redundant_columns = [f'redundant_feature_{i}' for i in range(n_redundant)]
    noisy_columns = [f'noisy_feature_{i}' for i in range(n_noisy)]
    df = pd.DataFrame(X_combined, columns=continuous_columns + discrete_columns + redundant_columns + noisy_columns)

    # Convert features to text if requested
    if textual_discrete:
        # Convert discrete features
        for idx, col in enumerate(discrete_columns):
            df[col] = df[col].apply(lambda x: discrete_word_mappings[idx][int(x)])
        
        # Convert redundant features
        for idx, col in enumerate(redundant_columns):
            # Normalize to bins and convert to text
            values = pd.Series(X_redundant[:, idx])
            binned = pd.qcut(values, n_bins, labels=False, duplicates='drop')
            df[col] = binned.map(lambda x: redundant_word_mappings[idx][int(x)])
        
        # Convert noisy features
        for idx, col in enumerate(noisy_columns):
            # Normalize to bins and convert to text
            values = pd.Series(X_noisy[:, idx])
            binned = pd.qcut(values, n_bins, labels=False, duplicates='drop')
            df[col] = binned.map(lambda x: noisy_word_mappings[idx][int(x)])
    else:
        for col in discrete_columns:
            df[col] = df[col].astype(int)

    df['label'] = y

    return df

if __name__ == "__main__":
    # Example usage
    n_samples = 1000
    n_continuous_features = 0
    n_discrete_features = 5
    n_classes = 5
    class_distribution = [0.1, 0.2, 0.3, 0.2, 0.2]
    n_bins = 5
    n_redundant = 10
    n_noisy = 20
    class_sep = 0.1

    # Generate dataset with numerical discrete features
    df_num = generate_synthetic_dataset(n_samples, n_continuous_features, n_discrete_features, 
                                      n_classes, class_distribution, textual_discrete=False)
    print("Numerical discrete features:")
    print(df_num.head())
    df_num.to_csv('synthetic_dataset_numerical.csv', index=False)
    
    # Generate dataset with textual discrete features
    df_text = generate_synthetic_dataset(n_samples, n_continuous_features, n_discrete_features, 
                                       n_classes, class_distribution, textual_discrete=True)
    print("\nTextual discrete features:")
    print(df_text.head())
    df_text.to_csv('synthetic_dataset_textual.csv', index=False)
