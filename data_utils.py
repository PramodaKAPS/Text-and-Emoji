# data_utils.py
import pandas as pd
from datasets import load_dataset, Dataset
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import emoji  # For emoji handling
import gensim.downloader as api  # For embeddings
import nltk  # For sentence tokenization and length checks
nltk.download('punkt', quiet=True)

# Load pre-trained word embeddings
word_vectors = api.load("glove-wiki-gigaword-300")

def preprocess_text(text, use_advanced_emoji=True, handle_short_sentences=True):
    """Advanced preprocessing: Handle emojis, short sentences, complex contexts, and punctuation."""
    # Emoji handling
    text = emoji.demojize(text, delimiters=("", ""))
    if use_advanced_emoji:
        # Semantic mappings for common emojis (expand as needed)
        mappings = {
            "smiling_face": "happy",
            "angry_face": "angry",
            "crying_face": "sad",
            "fearful_face": "scared",
            "surprised_face": "surprised"
        }
        for emoji_desc, word in mappings.items():
            text = text.replace(emoji_desc, word)
    
    # Punctuation handling for questions vs. statements
    if '?' in text:
        text = text.replace('?', ' [QUESTION]')
    else:
        text += ' [STATEMENT]'
    
    # Short sentence handling: Augment if <50 tokens
    if handle_short_sentences:
        tokens = nltk.word_tokenize(text)
        if len(tokens) < 50:
            text = "Short context: " + text  # Minimal framing for better model understanding
    
    # For complex sentences: No explicit change needed, as Longformer handles them via long max_length
    return text

def load_and_filter_goemotions(cache_dir, selected_emotions, num_train=0, use_advanced_emoji=True, handle_short_sentences=True):
    """
    Load and filter the GoEmotions dataset with advanced preprocessing.
    """
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)

    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    emotion_names = dataset["train"].features["labels"].feature.names
    selected_indices = [emotion_names.index(e) for e in selected_emotions if e in emotion_names]

    print("Selected emotions:", selected_emotions)
    print("Selected indices:", selected_indices)

    def filter_emotions(df):
        df = df.copy()
        df["labels"] = df["labels"].apply(lambda x: [label for label in x if label in selected_indices])
        df = df[df["labels"].apply(lambda x: len(x) > 0)]
        return df

    train_df = filter_emotions(train_df)
    valid_df = filter_emotions(valid_df)
    test_df = filter_emotions(test_df)

    train_df["label"] = train_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)
    valid_df["label"] = valid_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)
    test_df["label"] = test_df["labels"].apply(lambda lbls: lbls[0] if lbls else -1)

    train_df = train_df[train_df["label"] != -1]
    valid_df = valid_df[valid_df["label"] != -1]
    test_df = test_df[test_df["label"] != -1]

    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    # Apply advanced preprocessing
    train_df["text"] = train_df["text"].apply(lambda t: preprocess_text(t, use_advanced_emoji, handle_short_sentences))
    valid_df["text"] = valid_df["text"].apply(lambda t: preprocess_text(t, use_advanced_emoji, handle_short_sentences))
    test_df["text"] = test_df["text"].apply(lambda t: preprocess_text(t, use_advanced_emoji, handle_short_sentences))

    if num_train > 0:
        train_df = train_df.head(num_train)

    print(f"Filtered train data shape: {train_df.shape}")
    print(f"Filtered validation data shape: {valid_df.shape}")
    print(f"Filtered test data shape: {test_df.shape}")
    
    if train_df.empty:
        raise ValueError("Filtered train data is empty. Check selected emotions match dataset labels.")
    
    return train_df, valid_df, test_df, selected_indices

def oversample_training_data(train_df):
    """
    Oversample the training data to balance emotion classes using RandomOverSampler.
    """
    X = train_df["text"].values.reshape(-1, 1)
    y = train_df["label"]
    ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    df_resampled = pd.DataFrame({"text": X_resampled.flatten(), "label": y_resampled})
    print("Oversampled training data distribution:")
    print(df_resampled["label"].value_counts())
    return df_resampled

def prepare_tokenized_datasets(tokenizer, train_df, valid_df, test_df):
    """
    Tokenize datasets for advanced models with longer context (up to 4096 for complex sentences).
    """
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=4096)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_valid = valid_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)

    return tokenized_train, tokenized_valid, tokenized_test
