import nltk
import os
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import gutenberg, reuters
from nltk.tokenize import word_tokenize
import pandas as pd

class DatasetLoader:
    def __init__(self, target_tokens=1000000):
        self.target_tokens = target_tokens

    def load_gutenberg(self, output_file="corpus.txt"):
        nltk.download('gutenberg')
        sentences = gutenberg.sents()
        corpus = [word.lower() for sentence in sentences for word in word_tokenize(" ".join(sentence))][:self.target_tokens]
        self._save_corpus(corpus, output_file)
        return output_file

    def load_reuters(self, output_file="corpus.txt"):
        documents = reuters.fileids()
        corpus = [reuters.raw(doc_id) for doc_id in documents]
        self._save_corpus(corpus, output_file)
        return output_file

    def load_commoncrawl(self, output_file="corpus.txt"):
        commoncrawl_url = "https://commoncrawl.org/the-data/get-started/"
        response = requests.get(commoncrawl_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'common-crawl/crawl-data/CC-MAIN-\d{4}-\d{2}/segments/\d+/'))
        first_link = links[0]['href']
        corpus_url = f"https://commoncrawl.s3.amazonaws.com/{first_link}warc.paths.gz"
        response = requests.get(corpus_url)
        corpus = response.text.split('\n')[:self.target_tokens]  # Adjust the number of lines as needed
        self._save_corpus(corpus, output_file)
        return output_file
    def load_simlex_dataset(self, simlex_path):
        simlex_df = pd.read_csv(simlex_path, delimiter='\t')
        return simlex_df

    def _save_corpus(self, corpus, output_file):
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(" ".join(corpus))
    

