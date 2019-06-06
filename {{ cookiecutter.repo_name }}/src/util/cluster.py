"""
script use to do one-shot clustering
right now have kmeans and tokenizer for file paths
"""
import string
import collections

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from collections import Counter


def path_tokenize(path):
    return path.split("\\")


def path_tokenize2(path):
    raw_tokens = set(path.split("\\"))

    # removing disk path prefix
    tokens = []
    for token in raw_tokens:
        if len(token) == 2 and token[1] == ':':
            continue
        else:
            tokens.append(token.lower())
    return tokens


def path_tokenize3(path):
    prefixes = {'program files', 'program files (x86)', 'programdata', 'common files', 'system32', 'users', ''}
    # remove common path prefixes
    raw_tokens = set(path.split("\\"))
    raw_tokens = raw_tokens - prefixes

    # removing disk path prefix
    tokens = []
    for token in raw_tokens:
        if len(token) == 2 and token[1] == ':':
            continue
        else:
            tokens.append(token.lower())
    return tokens


# to do, parametrize tokenizer
def process_text(text, stem=False):
    """ Tokenize text and stem words removing punctuation """
    # text = text.translate(None, string.punctuation)
    tokens = path_tokenize3(text)

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


# to do, make vectorize a parameter (well then I may as well rebuild the whole pipeline
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.1,
                                 min_df=0.005,
                                 lowercase=True)

    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)

    clustering = collections.defaultdict(list)

    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    return clustering


def get_cluster_words(cluster_paths, n):
    file_paths = cluster_paths.apply(path_tokenize2).to_list()
    path_token_counter = Counter()
    for path in file_paths:
        path_token_counter.update(path)
    return path_token_counter.most_common(n)
