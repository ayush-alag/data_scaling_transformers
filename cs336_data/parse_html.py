from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text

def extract_text_from_html(bytes_str):
    encoding = detect_encoding(bytes_str)
    decoded_str = bytes_str.decode(encoding)
    return extract_plain_text(decoded_str)

def identify_language(text):
    # return a language code and a score
    raise NotImplementedError