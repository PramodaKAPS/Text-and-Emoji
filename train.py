# train.py

import os
from transformers import AutoTokenizer
from data_utils import load_and_filter_goemotions, oversample_training_data, prepare_tokenized_datasets
from model_utils import create_tf_datasets, setup_model_and_optimizer, compile_and_train, save_model_and_tokenizer, evaluate_model

def main():
    cache_dir = "/root/huggingface_cache"
    save_path = "/root/emotion_model"
    emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "neutral"]
    
    config = {
        "num_train": 0,  # Full dataset
        "num_epochs": 6,
        "batch_size": 8,
        "learning_rate": 5e-6
    }
    
    print("🚀 Starting Emotion Detection Training with Advanced Handling")
    print("=" * 60)
    print(f"📊 Training Configuration:")
    print(f"   - Cache directory: {cache_dir}")
    print(f"   - Save path: {save_path}")
    print(f"   - Selected emotions: {emotions}")
    print(f"   - Training samples: Full dataset")
    print(f"   - Epochs: {config['num_epochs']}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Learning rate: {config['learning_rate']}")
    print("-" * 60)
    
    try:
        train_df, valid_df, test_df, sel_indices = load_and_filter_goemotions(
            cache_dir, emotions, config["num_train"], use_advanced_emoji=True, handle_short_sentences=True
        )
        oversampled_train_df = oversample_training_data(train_df)
        
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir=cache_dir)
        tokenized_train, tokenized_valid, tokenized_test = prepare_tokenized_datasets(tokenizer, oversampled_train_df, valid_df, test_df)
        
        tf_train, tf_val, tf_test = create_tf_datasets(tokenized_train, tokenized_valid, tokenized_test, tokenizer, sel_indices, config["batch_size"])
        
        model, optimizer = setup_model_and_optimizer("allenai/longformer-base-4096", len(emotions), tf_train, config["num_epochs"], config["learning_rate"], cache_dir)
        
        model = compile_and_train(model, optimizer, tf_train, tf_val, config["num_epochs"])
        
        save_model_and_tokenizer(model, tokenizer, save_path)
        
        metrics = evaluate_model(model, tf_test, emotions)
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Final test results: {metrics}")
        print(f"💾 Model saved to: {save_path}")
        
    except Exception as e:
        print(f" Training failed: {e}")
        raise

if __name__ == "__main__":
    main()


