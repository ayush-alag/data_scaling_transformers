import nltk
from fasttext import load_model

def classify_given_fasttext_path(text, fasttext_path):
    model = load_model(fasttext_path)
    label, probability = model.predict(text.replace("\n", ""))
    return label[0].replace("__label__", ""), probability[0]

def classify_language(text):
    # return a language code and a score
    fastext_path = "/data/classifiers/lid.176.bin"
    return classify_given_fasttext_path(text, fastext_path)

def classify_nsfw(text):
    # return label (nsfw or not), and confidence score
    nsfw_classifier_path = "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
    return classify_given_fasttext_path(text, nsfw_classifier_path)

def classify_toxic(text):
    # return label (nsfw or not), and confidence score
    hate_classifier_path = "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
    return classify_given_fasttext_path(text, hate_classifier_path)

def classify_quality(text):
    # high quality or not + confidence score
    trained_qc_path = "/data/c-aalag/processed_classifier_data/quality_classifier/quality_classifier.bin"
    label, score = classify_given_fasttext_path(text, trained_qc_path)
    if label == "negative":
        return "cc", score
    if label == "positive":
        return "wiki", score

def gopher_quality_filter(text):
    # return bool of passing gopher quality or not
    nltk.download('punkt_tab', quiet=True)
    words = nltk.word_tokenize(text)

    if len(words) < 50 or len(words) > 100000:
        return False

    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # check how many words have an alphabetic character
    alpha_words = [word for word in words if any(c.isalpha() for c in word)]
    if len(alpha_words) / len(words) < 0.8:
        return False

    # check how many lines end with "..." ellipsis
    lines = text.split("\n")
    ellipsis_lines = [line for line in lines if line.endswith("...")]
    if len(ellipsis_lines) / len(lines) > 0.3:
        return False

    return True
