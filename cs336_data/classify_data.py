import nltk

def classify_nsfw(text):
    # return label (nsfw or not), and confidence score
    raise NotImplementedError

def classify_toxic(text):
    # return label (nsfw or not), and confidence score
    raise NotImplementedError

def classify_quality(text):
    # high quality or not + confidence score
    raise NotImplementedError

def gopher_quality_filter(text):
    # return bool of passing gopher quality or not
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
