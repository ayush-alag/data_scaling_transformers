import importlib.metadata
from cs336_data.parse_html import warc_text_iterator, extract_text_from_html
from cs336_data.classify_data import LanguageClassifier, NSFWClassifier, ToxicClassifier, GopherQualityClassifier

__version__ = importlib.metadata.version("cs336-data")