import numpy as np
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import emoji
import gensim.downloader as api
import nltk
nltk.download('punkt', quiet=True)

word_vectors = api.load("glove-wiki-gigaword-300")

class EmotionDetector:
    def __init__(self, model_path, emotions_list):
        self.emotions = emotions_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        print(f"✅ Emotion detector loaded from {model_path}")

    def preprocess_with_advanced_handling(self, text):
        text = emoji.demojize(text, delimiters=("", ""))
        mappings = {
            "smiling_face": "happy",
            "angry_face": "angry",
            "crying_face": "sad",
            "fearful_face": "scared",
            "surprised_face": "surprised"
        }
        for emoji_desc, word in mappings.items():
            text = text.replace(emoji_desc, word)
        if '?' in text:
            text = text.replace('?', ' [QUESTION]')
        else:
            text += ' [STATEMENT]'
        tokens = nltk.word_tokenize(text)
        if len(tokens) < 50:
            text = "Short context: " + text
        return text

    def predict_emotion(self, text):
        text = self.preprocess_with_advanced_handling(text)
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
        logits = self.model(inputs).logits
        prediction = np.argmax(logits, axis=1)[0]
        return self.emotions[prediction]
    
    def predict_emotion_with_confidence(self, text):
        text = self.preprocess_with_advanced_handling(text)
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
        logits = self.model(inputs).logits
        import tensorflow as tf
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        emotion_scores = dict(zip(self.emotions, [float(prob) for prob in probabilities]))
        predicted_emotion = self.emotions[np.argmax(probabilities)]
        return {
            "predicted_emotion": predicted_emotion,
            "confidence": float(probabilities.max()),
            "all_scores": emotion_scores
        }

def interactive_emotion_detection(model_path, emotions_list):
    detector = EmotionDetector(model_path, emotions_list)
    print("\n🎭 Interactive Emotion Detection with Advanced Handling")
    print("Enter text to analyze emotions (press Enter to exit)")
    print("-" * 50)
    while True:
        text = input("\nEnter text: ")
        if text.strip() == "":
            print("👋 Goodbye!")
            break
        try:
            result = detector.predict_emotion_with_confidence(text)
            print(f"🎯 Predicted emotion: {result['predicted_emotion']}")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            sorted_emotions = sorted(
                result['all_scores'].items(), key=lambda x: x[1], reverse=True
            )[:3]
            print("📈 Top 3 emotions:")
            for emotion, score in sorted_emotions:
                print(f"   {emotion}: {score:.3f}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    model_path = "/root/emotion_model"
    emotions_list = [
        "anger", "sadness", "joy", "disgust", "fear",
        "surprise", "neutral"
    ]
    interactive_emotion_detection(model_path, emotions_list)

