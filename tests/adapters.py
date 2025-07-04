from __future__ import annotations
from cs336_data.parse_html import extract_text_from_html
from cs336_data.classify_data import LanguageClassifier, NSFWClassifier, QualityClassifier, ToxicClassifier, GopherQualityClassifier
from cs336_data.mask_data import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.deduplication import exact_line_deduplication, minhash_deduplication

import os
from typing import Any



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return LanguageClassifier().classify(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return NSFWClassifier().classify(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return ToxicClassifier().classify(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return QualityClassifier().classify(text)


def run_gopher_quality_filter(text: str) -> bool:
    return GopherQualityClassifier().classify(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplication(input_files,
                                 num_hashes,
                                 num_bands,
                                 ngrams,
                                 jaccard_threshold,
                                 output_directory)
