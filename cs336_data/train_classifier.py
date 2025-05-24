import gzip
import random
import subprocess
import warc
from bs4 import BeautifulSoup
import fasttext
import os
import tempfile
import requests
from typing import Tuple, List
import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from classify_data import gopher_quality_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityClassifierPipeline:
    def __init__(self, wiki_urls_path: str, cc_data_path: str = '/data/CC/',
                 model_save_path: str = 'quality_model.bin'):
        self.wiki_urls_path = wiki_urls_path
        self.cc_data_path = cc_data_path
        self.model_save_path = model_save_path
        self.model = None

    def subsample_urls(self, n_samples: int = 20000) -> List[str]:
        """Randomly subsample URLs from Wikipedia dump."""
        logger.info(f"Subsampling {n_samples} URLs from Wikipedia dump...")

        # First pass: count total URLs for proper random sampling
        total_urls = 0
        with gzip.open(self.wiki_urls_path, 'rt') as f:
            for line in f:
                if line.strip().startswith('http'):
                    total_urls += 1

        logger.info(f"Total URLs available: {total_urls}")

        # Calculate sampling probability
        sample_prob = min(1.0, (n_samples * 3) / total_urls)  # 3x for filtering headroom

        urls = []
        with gzip.open(self.wiki_urls_path, 'rt') as f:
            for line in f:
                url = line.strip()
                if url.startswith('http') and random.random() < sample_prob:
                    # Filter out non-text content
                    if not any(ext in url.lower() for ext in
                             ['.pdf', '.doc', '.xls', '.ppt', '.zip', '.jpg',
                              '.png', '.gif', '.mp4', '.mp3', '.mov']):
                        urls.append(url)

                if len(urls) >= n_samples * 2:  # 2x for safety
                    break

        # Final random sample to exact size
        return random.sample(urls, min(n_samples, len(urls)))

    def fetch_url_text(self, url: str, timeout: int = 5) -> str:
        """Fetch and extract text from a single URL."""
        try:
            response = requests.get(url, timeout=timeout, headers={
                'User-Agent': 'Mozilla/5.0 (Compatible; QualityClassifier/1.0)'
            })
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return ""

            return extract_text_from_html(response.content)
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return ""

    def collect_positive_examples(self, n_samples: int = 10000) -> List[str]:
        """Collect positive examples from Wikipedia-linked sources."""
        logger.info(f"Collecting {n_samples} positive examples...")

        urls = self.subsample_urls(n_samples)
        positive_texts = []

        # Use parallel fetching for speed
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_url = {executor.submit(self.fetch_url_text, url): url
                           for url in urls}

            with tqdm(total=len(urls), desc="Fetching positive examples") as pbar:
                for future in as_completed(future_to_url):
                    text = future.result()
                    pbar.update(1)

                    # Quality checks
                    if gopher_quality_filter(text):
                        positive_texts.append(text)

                    if len(positive_texts) >= n_samples:
                        break

        logger.info(f"Collected {len(positive_texts)} positive examples")
        return positive_texts[:n_samples]

    def extract_from_common_crawl(self, n_samples: int = 10000) -> List[str]:
        """Extract random samples from Common Crawl data."""
        logger.info(f"Extracting {n_samples} samples from Common Crawl...")

        cc_files = list(Path(self.cc_data_path).glob('*.warc.gz'))
        if not cc_files:
            logger.error(f"No WARC files found in {self.cc_data_path}")
            return []

        texts = []
        samples_per_file = max(1, n_samples // len(cc_files))

        for cc_file in tqdm(cc_files[:10], desc="Processing CC files"):  # Limit files
            try:
                file_texts = self._extract_from_warc(cc_file, samples_per_file)
                texts.extend(file_texts)

                if len(texts) >= n_samples:
                    break

            except Exception as e:
                logger.error(f"Error processing {cc_file}: {e}")
                continue

        return texts[:n_samples]

    def _extract_from_warc(self, warc_path: Path, max_samples: int) -> List[str]:
        """Extract text samples from a WARC file."""
        texts = []

        with gzip.open(warc_path, 'rb') as f:
            for record in warc.WARCFile(fileobj=f):
                if len(texts) >= max_samples:
                    break

                if record.rec_type == 'response':
                    try:
                        # Get content
                        content = record.content_stream().read()

                        # Check if HTML
                        if b'<html' not in content.lower()[:1000]:
                            continue

                        # Parse HTML
                        soup = BeautifulSoup(content, 'html.parser')

                        # Remove scripts and styles
                        for element in soup(['script', 'style']):
                            element.decompose()

                        text = soup.get_text(separator=' ', strip=True)

                        # Basic length check
                        if len(text) > 100 and len(text.split()) > 50:
                            texts.append(text)

                    except Exception:
                        continue

        return texts

    def collect_negative_examples(self, n_samples: int = 10000) -> List[str]:
        """Collect negative examples from Common Crawl."""
        logger.info("Collecting negative examples from Common Crawl...")

        # Get random Common Crawl samples
        cc_texts = self.extract_from_common_crawl(n_samples * 2)  # Extra for filtering

        negative_texts = []
        for text in cc_texts:
            # Apply inverse of quality criteria - we want "random" quality texts
            if not gopher_quality_filter(text):
                negative_texts.append(text)
            elif random.random() < 0.3:  # Include some medium quality as negative
                negative_texts.append(text)

            if len(negative_texts) >= n_samples:
                break

        # If we don't have enough, generate some synthetic ones
        if len(negative_texts) < n_samples:
            logger.info("Generating additional synthetic negative examples...")
            synthetic = self.generate_synthetic_negatives(n_samples - len(negative_texts))
            negative_texts.extend(synthetic)

        logger.info(f"Collected {len(negative_texts)} negative examples")
        return negative_texts[:n_samples]

    def prepare_training_data(self, positive_texts: List[str],
                            negative_texts: List[str],
                            val_split: float = 0.1) -> Tuple[str, str]:
        """Prepare data in fastText format with train/val split."""
        logger.info("Preparing training data...")

        # Combine and shuffle
        all_data = []
        for text in positive_texts:
            cleaned = ' '.join(text.split()[:1000])  # Limit length
            cleaned = cleaned.replace('\n', ' ').replace('\t', ' ')
            all_data.append(('__label__high', cleaned))

        for text in negative_texts:
            cleaned = ' '.join(text.split()[:1000])
            cleaned = cleaned.replace('\n', ' ').replace('\t', ' ')
            all_data.append(('__label__low', cleaned))

        # Shuffle
        random.shuffle(all_data)

        # Split
        val_size = int(len(all_data) * val_split)
        val_data = all_data[:val_size]
        train_data = all_data[val_size:]

        # Write training file
        train_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for label, text in train_data:
            train_file.write(f'{label} {text}\n')
        train_file.close()

        # Write validation file
        val_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for label, text in val_data:
            val_file.write(f'{label} {text}\n')
        val_file.close()

        return train_file.name, val_file.name

    def train_classifier(self, training_file: str, validation_file: str):
        """Train the fastText classifier with validation."""
        logger.info("Training classifier...")

        # Train the model with auto-tuning
        self.model = fasttext.train_supervised(
            training_file,
            autotuneValidationFile=validation_file,
            autotuneDuration=600,  # 10 minutes
            epoch=25,
            lr=0.5,
            wordNgrams=2,
            dim=100,
            loss='softmax',
            thread=8,  # Use more threads on cluster
            verbose=2
        )

        # Test on validation set
        n_correct, n_total = self.model.test(validation_file)
        logger.info(f"Validation accuracy: {n_correct/n_total:.2%}")

        # Save the model
        self.model.save_model(self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")

        # Clean up
        os.unlink(training_file)
        os.unlink(validation_file)

    def classify(self, text: str) -> Tuple[bool, float]:
        """Classify a single text."""
        if not self.model:
            self.model = fasttext.load_model(self.model_save_path)

        # Preprocess
        cleaned = ' '.join(text.split()[:1000])  # Same preprocessing as training
        cleaned = cleaned.replace('\n', ' ').replace('\t', ' ')

        # Predict
        labels, probs = self.model.predict(cleaned, k=2)

        # Get probabilities for both classes
        label_probs = {}
        for label, prob in zip(labels, probs):
            label_probs[label] = prob

        # Get probability of being high quality
        high_prob = label_probs.get('__label__high', 0.0)

        # Decision based on probability
        is_high_quality = high_prob > 0.5
        confidence = high_prob if is_high_quality else (1 - high_prob)

        return (is_high_quality, confidence)

    def train_full_pipeline(self, n_positive: int = 10000, n_negative: int = 10000):
        """Run the full training pipeline."""
        # Collect data
        positive_texts = self.collect_positive_examples(n_positive)
        negative_texts = self.collect_negative_examples(n_negative)

        logger.info(f"Final dataset: {len(positive_texts)} positive, {len(negative_texts)} negative")

        # Prepare training data
        training_file, validation_file = self.prepare_training_data(
            positive_texts, negative_texts, val_split=0.1
        )

        # Train classifier
        self.train_classifier(training_file, validation_file)

        # Test on some examples
        self.test_classifier()

    def test_classifier(self):
        """Test the classifier on some examples."""
        test_examples = [
            ("This is a well-written article about machine learning with proper citations and clear explanations. The article discusses various algorithms and their applications in detail.", True),
            ("BUY NOW!!! CLICK HERE FOR FREE STUFF!!! LIMITED TIME OFFER!!! DISCOUNT PILLS VIAGRA!!!", False),
            ("The quantum mechanical model of the atom describes electrons as wave functions rather than particles in fixed orbits. This revolutionary understanding came from the work of Schrödinger and Heisenberg.", True),
            ("spam spam spam spam spam spam spam spam spam spam click here buy now free offer", False),
            ("Error 404: Page not found. The requested URL was not found on this server.", False),
        ]

        print("\nTesting classifier on examples:")
        print("-" * 80)

        correct = 0
        for text, expected in test_examples:
            is_high_quality, confidence = self.classify(text)
            correct += (is_high_quality == expected)

            print(f"Text: {text[:80]}...")
            print(f"Expected: {expected}, Predicted: {is_high_quality}, Confidence: {confidence:.3f}")
            print("-" * 80)

        print(f"\nTest accuracy: {correct}/{len(test_examples)} = {correct/len(test_examples):.1%}")


# Adapter function for pytest
def run_classify_quality(text: str) -> Tuple[bool, float]:
    """Classify text quality - adapter for pytest tests."""
    # Load pre-trained model
    classifier = QualityClassifierPipeline(
        wiki_urls_path='/data/wiki/enwiki-20240420-extracted_urls.txt.gz',
        cc_data_path='/data/CC/',
        model_save_path='quality_model.bin'
    )
    return classifier.classify(text)


# Training script
if __name__ == '__main__':
    # Initialize pipeline
    pipeline = QualityClassifierPipeline(
        wiki_urls_path='/data/wiki/enwiki-20240420-extracted_urls.txt.gz',
        cc_data_path='/data/CC/'
    )

    # Train the model - using 10k each for reasonable training time
    # Scale up to 50k+ for better performance
    pipeline.train_full_pipeline(n_positive=10000, n_negative=10000)