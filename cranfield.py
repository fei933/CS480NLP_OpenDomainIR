from inforet import SpacyInfoRet
from inforet import InfoRet, Query, Document
from io import TextIOBase
from typing import Type
from sys import argv
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

def parse_cran_docs(path: Path, instance: InfoRet):
    with open(path, "r") as inp:
        current_doc = ""
        current_doc_id = 0
        for line in inp.readlines():
            if not line:
                continue
            if line[0:2] == ".I":
                # finish the document and start a new one
                if current_doc and current_doc_id:
                    instance.add_document(current_doc_id, current_doc)
                current_doc = ""
                current_doc_id = int(line[3:])
            elif line[0] == ".":
                current_doc += line[3:]
            else:
                current_doc += line
        # submit the final doc when there are no more lines
        instance.add_document(current_doc_id, current_doc)


def parse_cran_queries(path: Path, instance: InfoRet) -> list[Query]:
    all_queries = []
    with open(path, "r") as inp:
        current_qry = ""
        current_query_id = 0
        for line in inp.readlines():
            if not line:
                continue
            if line[0:2] == ".I":
                # finish the current query and start a new one
                if current_query_id:
                    all_queries.append(instance.make_query(current_query_id, current_qry))
                current_qry = ""
                current_query_id += 1
            elif line[0:2] == ".W":
                current_qry += line[3:]
            else:
                current_qry += line
    all_queries.append(instance.make_query(current_query_id, current_qry))
    return all_queries

def query_and_print(instance: InfoRet, query: Query, out: TextIOBase):
    for (doc, score) in instance.perform_query(query):
        out.write(f"{query.ident} {doc.ident} {score:.4f}\n")


def run_cranqrel(
    documents_path: Path,
    queries_path: Path,
    output_path: Path,
    instance: InfoRet = InfoRet()
):
    parse_cran_docs(documents_path, instance)
    with open(output_path, "w") as out:
        for qry in parse_cran_queries(queries_path, instance):
            query_and_print(instance, qry, out)


if __name__ == "__main__":
    docs = Path(argv[1])
    queries = Path(argv[2])
    try:
        resultsdir = Path(argv[3])
    except IndexError:
        resultsdir = Path("results")

    resultsdir.mkdir(exist_ok = True)

    stopwords = set(class_stop_words)

    punct = set([".", ",", "'", '?', '!', ';', ':'])

    punct_stopwords = stopwords.union(punct)

    stemmer = EnglishStemmer()

    tests = [
      #  ("00_simple",
       #  InfoRet()),
        # one feature
   #     ("01_downcase",
   #      InfoRet(downcase = True)),
    #    ("02_punct",
     #    InfoRet(stopwords = punct)),
        ("03_nltkstopwords",
         InfoRet(stopwords = stopwords)),
 #       ("04_snowballstemmer",
#         InfoRet(stemmer = stemmer)),
        # two features, w/ downcase
   #    ("05_downcase_punct",
   #      InfoRet(downcase = True, stopwords = punct)),
   #     ("06_downcase_nltkstopwords",
  #       InfoRet(stopwords = stopwords, downcase = True)),
 #       ("07_downcase_snowballstemmer",
#         InfoRet(downcase = True, stemmer = stemmer)),
         #two features, w/ punct
        ("08_punct_nltkstopwords",
         InfoRet(stopwords = punct_stopwords)),
     #   ("09_punct_snowballstemmer",
     #    InfoRet(stopwords = punct, stemmer = stemmer)),
        # two features, w/ nltkstopwords
        ("10_nltkstopwords_snowballstemmer",
         InfoRet(stopwords = stopwords, stemmer = stemmer)),
         #three features, not snowball
    #    ("11_downcase_punct_nltkstopwords",
     #    InfoRet(downcase = True, stopwords = punct_stopwords)),
        # three features, not punct
       # ("12_downcase_nltkstopwords_snowballstemmer",
      #   InfoRet(downcase = True, stemmer = stemmer, stopwords = stopwords)),
        # three features, not nltkstopwords
    #    ("13_downcase_punct_snowballstemmer",
     #    InfoRet(downcase = True, stopwords = punct, stemmer = stemmer)),
        # three features, not downcase
        ("14_punct_nltkstopwords_snowballstemmer",
         InfoRet(stopwords = punct_stopwords, stemmer = stemmer)),
        # all four
        ("15_downcase_punct_nltkstopwords_snowballstemmer",
         InfoRet(downcase = True, stopwords = punct_stopwords, stemmer = stemmer)),
        #Spacy stuff
    #    ("16_spacy_full_normalization",
     #    SpacyInfoRet(stopwords = True, stemmer = True, punct = True, use_vector = 0)),
        #Spacy punct and stem
     #  ("17_spacy_punct_stem",
      #   SpacyInfoRet(stopwords = False, stemmer = True, punct = True, use_vector = 0)),
        #Spacy stop and stem
#        ("18_spacy_stop_stem",
 #        SpacyInfoRet(stopwords = True, stemmer = True, punct = False, use_vector = 0)),
        #Spacy stop and punct
  #      ("19_spacy_stop_punct",
   #      SpacyInfoRet(stopwords = True, stemmer = False, punct = True, use_vector = 0)),
        #Spacy stem
    #    ("20_spacy_stem",
     #    SpacyInfoRet(stopwords = False, stemmer = True, punct = False, use_vector = 0)),
        #Spacy stop
      #  ("21_spacy_stop",
       #  SpacyInfoRet(stopwords = True, stemmer = False, punct = False, use_vector = 0)),
         #Same but all w wordvec
    #    ("22_spacy_full_normalization_wordvec",
     #    SpacyInfoRet(stopwords = True, stemmer = True, punct = True, use_vector = 1)),
        #Spacy punct and stem
  #      ("23_spacy_punct_stem_wordvec",
   #      SpacyInfoRet(stopwords = False, stemmer = True, punct = True, use_vector = 1)),
        #Spacy stop and stem
    #    ("24_spacy_stop_stem_wordvec",
     #    SpacyInfoRet(stopwords = True, stemmer = True, punct = False, use_vector = 1)),
        #Spacy stop and punct
 #       ("25_spacy_stop_punct_wordvec",
  #       SpacyInfoRet(stopwords = True, stemmer = False, punct = True, use_vector = 1)),
        #Spacy stem
   #     ("26_spacy_stem_wordvec",
    #     SpacyInfoRet(stopwords = False, stemmer = True, punct = False, use_vector = 1)),
        #Spacy stop
     #   ("27_spacy_stop_wordvec",
      #   SpacyInfoRet(stopwords = True, stemmer = False, punct = False, use_vector = 1)),
         #Same but all with wordvecnorm
 #       ("28_spacy_full_normalization_wordvecnorm",
  #       SpacyInfoRet(stopwords = True, stemmer = True, punct = True, use_vector = 2)),
        #Spacy punct and stem
 #       ("29_spacy_punct_stem_wordvecnorm",
  #       SpacyInfoRet(stopwords = False, stemmer = True, punct = True, use_vector = 2)),
        #Spacy stop and stem
 #       ("30_spacy_stop_stem_wordvecnorm",
  #       SpacyInfoRet(stopwords = True, stemmer = True, punct = False, use_vector = 2)),
        #Spacy stop and punct
  #      ("31_spacy_stop_punct_wordvecnorm",
   #      SpacyInfoRet(stopwords = True, stemmer = False, punct = True, use_vector = 2)),
        #Spacy stem
 #       ("32_spacy_stem_wordvecnorm",
  #       SpacyInfoRet(stopwords = False, stemmer = True, punct = False, use_vector = 2)),
        #Spacy stop
 #       ("33_spacy_stop_wordvecnorm",
  #       SpacyInfoRet(stopwords = True, stemmer = False, punct = False, use_vector = 2)),
    ]

    for (name, instance) in tests:
        print(f"running {name}")
        run_cranqrel(
            documents_path = docs,
            queries_path = queries,
            output_path = resultsdir / name,
            instance = instance,
        )
