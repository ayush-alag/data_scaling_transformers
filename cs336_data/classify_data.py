import nltk
from fasttext import load_model

def classify_given_fasttext_path(text, model):
    label, probability = model.predict(text.replace("\n", ""))
    return label[0].replace("__label__", ""), probability[0]

class LanguageClassifier:
    def __init__(self):
        self.model = load_model("/data/classifiers/lid.176.bin")

    def classify(self, text):
        return classify_given_fasttext_path(text, self.model)

class NSFWClassifier:
    def __init__(self):
        self.model = load_model("/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")

    def classify(self, text):
        return classify_given_fasttext_path(text, self.model)

class QualityClassifier:
    def __init__(self):
        self.model = load_model("/data/c-aalag/processed_classifier_data/quality_classifier/quality_classifier.bin")

    def classify(self, text):
        label, score = classify_given_fasttext_path(text, self.model)
        if label == "negative":
            return "cc", score
        if label == "positive":
            return "wiki", score

class ToxicClassifier:
    def __init__(self):
        self.model = load_model("/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

    def classify(self, text):
        return classify_given_fasttext_path(text, self.model)

class GopherQualityClassifier:
    def __init__(self):
        nltk.download('punkt_tab', quiet=True)

    def classify(self, text):
        words = nltk.word_tokenize(text)

        if len(words) < 50 or len(words) > 100000:
            return False

        # dont want too small or too large words
        mean_word_length = sum(len(word) for word in words) / len(words)
        if mean_word_length < 3 or mean_word_length > 10:
            return False

        # how many words have an alphabetic character
        alpha_words = [word for word in words if any(c.isalpha() for c in word)]
        if len(alpha_words) / len(words) < 0.8:
            return False

        # how many lines end with ellipsis
        lines = text.split("\n")
        ellipsis_lines = [line for line in lines if line.endswith("...")]
        if len(ellipsis_lines) / len(lines) > 0.3:
            return False

        return True