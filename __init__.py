import numpy as np
import numpy.linalg as linalg
import logging
from typing import Type, Iterator, Optional
import nltk
from nltk.stem.api import StemmerI
from collections import Counter
import math
import xml.etree.ElementTree as ET
import spacy
from spacy.tokens import Doc
log = logging.getLogger(__name__)

#Owen stuff. Takes spacy doc and normalizes it, returns as string
nlp = spacy.load("en_core_web_lg")

def normalize_spacy_doc(doc: Doc, stem: bool, stop: bool, punct: bool):
    if stem:
        lemmas = [nlp.vocab[token.lemma_] for token in doc]
    else:
        lemmas = [token for token in doc]

    if stop:
        stopped = [token for token in lemmas
                   if not token.is_stop]
    else:
        stopped = lemmas

    if punct:
        punctuations="?:!.,;"
        depuncted = [token for token in stopped
                     if token.text not in punctuations]
    else:
        depuncted = stopped

    return [token.text for token in depuncted]

class Document:
    ident: int
    word_frequencies: Counter
    length: int

    def __init__(self, ident: int, text: list[str]):
        assert text
        self.ident = ident
        self.word_frequencies = Counter(text)
        self.length = len(text)


class Query(Document):
    def unique_tokens(self) -> Iterator[str]:
        # Counter has a consistent iteration order
        return self.word_frequencies.elements()


class InfoRet:
    document_class: Type[Document] = Document
    query_class: Type[Query] = Query

    stopwords: Optional[set[str]]
    stemmer: Optional[StemmerI]
    downcase: bool
    inclusion_threshold: int = 0.0001 # minimum similarity to include in results

    def __init__(
        self,
        *,
        stopwords = None,
        stemmer = None,
        downcase = False,
    ):
        self.documents: set[Document] = set()
        self.downcase = downcase
        self.stopwords = stopwords
        self.stemmer = stemmer
    
    def is_stopword(self, word: str) -> bool:
        if self.stopwords:
            return word in self.stopwords
        else:
            return False

    def normalize_word(self, word: str) -> Iterator[str]:
        if self.downcase:
            yield word.lower()
        else:
            yield word

    def tokenize(self, seq: str) -> list[str]:
        return nltk.word_tokenize(seq)

    def normalize_text(self, text: str) -> list[str]:
        return [norm_word for word in self.tokenize(text)
                for norm_word in self.normalize_word(word)
                if not self.is_stopword(norm_word)]

    def add_document(self, ident: int, text: str) -> Document:
        doc = self.document_class(ident, self.normalize_text(text))
        self.documents.add(doc)
        return doc

    def parse_document(self, path: str) -> Document:
        doc = ET.parse(path)
        text = ""
        for child in doc:
            text += child.text

    def make_query(self, ident: int, text: str) -> Query:
        return self.query_class(ident, self.normalize_text(text))

    def term_idf(self, term: str) -> float:
        freq = sum(doc.word_frequencies[term] for doc in self.documents)
        return math.log(len(self.documents) / (1 + freq))

    def document_term_freq(self, term: str, doc: Document) -> float:
        return doc.word_frequencies[term] / doc.length

    def query_idf_vector(self, query: Query) -> np.ndarray:
        return np.array([self.term_idf(term) for term in query.unique_tokens()])

    def document_tf_vector(self, query: Query, doc: Document) -> np.ndarray:
        return np.array([self.document_term_freq(term, doc)
                         for term
                         in query.unique_tokens()])

    def document_tf_idf_vector(
        self,
        doc: Document,
        query: Query,
        idfs: np.ndarray,
    ) -> np.ndarray:
        return self.document_tf_vector(query, doc) * idfs

    def query_all_document_vectors(
        self,
        query: Query,
        idfs: np.ndarray,
    ) -> dict[Document, np.ndarray]:
        return { doc: self.document_tf_idf_vector(doc, query, idfs) for doc in self.documents }

    def vector_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        # cosine similarity!
        return np.dot(left, right) / (linalg.norm(left) * linalg.norm(right))
        

    def sort_results(
        self,
        results: dict[Document, np.ndarray],
        query: Query
    ) -> list[tuple[Document, float]]:
        idfs = self.query_idf_vector(query)
        query_tfidf = self.document_tf_idf_vector(query, query, idfs)
        doc_vecs = self.query_all_document_vectors(query, idfs)
        tuples = []
        for (doc, doc_tfidf) in doc_vecs.items():
            score = self.vector_similarity(query_tfidf, doc_tfidf)
            if score > self.inclusion_threshold:
                tuples.append((doc, score))
        return sorted(tuples, key=lambda pair: pair[1], reverse = True)

    def perform_query(self, query: Query) -> list[tuple[Document, float]]:
        idfs = self.query_idf_vector(query)
        docs = self.query_all_document_vectors(query, idfs)
        return self.sort_results(docs, query)

#Subclassed some things to integrate spacy -Owen
class SpacyInfoRet(InfoRet):

    stopwords: bool 
    stemmer: bool
    punct: bool
    use_vector: int

    def __init__(
        self,
        *,
        stopwords = False,
        punct = False,
        stemmer = False,
        downcase = False,
        use_vector = 0

    ):
        self.documents: set[Document] = set()
        self.downcase = downcase
        self.stopwords = stopwords
        self.punct = punct
        self.stemmer = stemmer
        self.use_vector = use_vector

    def add_document(self, ident: int, text: str) -> Document:
        spacy_doc = nlp(text)
        doc = self.document_class(ident, normalize_spacy_doc(spacy_doc, self.stemmer, self.stopwords, self.punct))
        self.documents.add(doc)
        return doc

    def make_query(self, ident: int, text: str) -> Query:
        spacy_doc = nlp(text)
        return self.query_class(ident, normalize_spacy_doc(spacy_doc, self.stemmer, self.stopwords, self.punct))

    def sort_results(
        self,
        results: dict[Document, np.ndarray],
        query: Query
    ) -> list[tuple[Document, float]]:
        tuples = []
        if self.use_vector == 0:
            idfs = self.query_idf_vector(query)
            query_tfidf = self.document_tf_idf_vector(query, query, idfs)
            doc_vecs = self.query_all_document_vectors(query, idfs)
            for (doc, doc_tfidf) in doc_vecs.items():
                score = self.vector_similarity(query_tfidf, doc_tfidf)
                if score > self.inclusion_threshold:
                    tuples.append((doc, score))
        elif self.use_vector == 1:
            queryvec = self.text_vector(query)
            for doc in self.documents:
                docvec = self.text_vector(doc)
                score = self.vector_similarity(queryvec, docvec)
                if score > self.inclusion_threshold:
                    tuples.append((doc, score))
        else:
            queryvec = self.text_vector_norm(query)
            for doc in self.documents:
                docvec = self.text_vector_norm(doc)
                score = self.vector_similarity(queryvec, docvec)
                if score > self.inclusion_threshold:
                    tuples.append((doc, score))
        return sorted(tuples, key=lambda pair: pair[1], reverse = True)