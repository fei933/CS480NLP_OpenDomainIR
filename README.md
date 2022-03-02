# Modulated Information Retrieval Task on Domain-Specific Collections

**Author:** Owen Cozine, Phoebe Goldman, Julia Murillo, & Feifei Wang

**1 Abstract**

Information retrieval systems return documents from a collection that are relevant to a particular information query. The efficacy of a system’s subtasks (e.g. removal of stop words, normalization of capitalization, etc.) is greatly dependent on the language used in the collection of queries and documents. We attempt an ad-hoc information retrieval task on the Cranfield 1400 Collection and NFCorpus, completing the task successfully on the Cranfield corpus. Using well-documented evaluation measures, we compare the efficacy of various auxiliary, open-source, information retrieval subtasks on this English language corpus. We find the features implemented from spaCy, particularly the stopwords list, to be the most effective at producing relevant results in the Cranfield corpus.

**1 Introduction and Motivation**

Information retrieval has been a topic of interest in Natural Language Processing (NLP) for
decades. Formally, information retrieval consists of returning relatively small amounts of
relevant material from a larger collection of documents that satisfy a particular information need,
typically presented in the form of a question or query (van Rijsbergen, 1979). With the rise of the
Internet and the widespread use of search engines, the everyday person expects the answer to any
question that might cross their mind to be at their fingertips. These answers, however, lay amid
an increasingly large number of documents that need to be reviewed. Information retrieval tasks
serve to narrow down this number so that the user needs only review the most relevant
documents, and therefore have only become more relevant in the pursuit of optimizing user
experience in a world of ever-expanding user expectations. As large-scale systems, by necessity,
are not optimized for finding domain-specific information, smaller scale retrieval systems may
become more important in academic realms that require such specific relevant information, like
that of medicine (Pohorec et al., 2009). What material is relevant is entirely dependent on the
domain of the collection of documents being queried, as even in English-language academia (the
focus of this study), style and language differ greatly among different domains and countries of
origin. The word “function,” for instance, has different meanings in physical science (e.g. a
linear function of the distance around a cylinder) and nutrition (e.g. a patient’s artery function).
This study seeks to explore the efficacy of different features on a baseline information retrieval
system using two corpora from different academic domains: aerodynamics and nutrition.

Despite contemporary use of information retrieval for large-scale search engines, initial
research into information retrieval centered around sets of queries and documents that use
formal, domain-specific language. The pioneering Cranfield experiments completed a set of
information retrieval tasks on collections of abstracts from aerodynamics articles, developing
queries that also fall in the aerodynamics domain (Manning et al, 2009). This narrow focus
requires relatively fewer considerations when trying to determine what information might
actually be relevant to a particular query as compared to large-scale searches. Besides the
obvious fact that all abstracts and queries are centered around the same general topic, the task

was relatively simplified by a few other factors that come from using a domain-specific, formal
language collection, including: the high probability that terms used in queries are well-defined
within that particular domain, the likelihood that queries and documents developed in an
academic setting do not contain any spelling errors or other typos, and the unlikelihood of any
slang or idioms being used.

**2 State of the Art**

As mentioned above, in recent years there has been increased interest in domain-specific
language information retrieval. This is due in no small part to the continuously growing field of
medicine and medicine-related information that might be queried, including both academic
scientific research and patient information. Outlined below are some of the articles within this
burgeoning topic. While the present study examines well-documented methods in the context of
domain-specific language, these articles are on the cutting edge and introduce new methods of
information retrieval that can be tailored to particular domains.

In order to create an information retrieval system that is compatible with highly-specific
medical domains, Fautsch & Savoy (2010) suggest an adaptation of the vector-space model that
accounts for a term’s relevance in the target domain, among other adjustments. (The basics of the
vector-space model are described below, in Section 4.1.) The authors proposed that, since the
system is domain-specific, a term’s frequency within that corpus would be a relevant measure to
consider when determining the weight of that term. Their model was evaluated on four separate
domain-specific collections in English and German against the classic vector-space model and a
probabilistic mode; they used the classical vector-space model’s performance as a baseline
performance goal and the probabilistic model as an upper-level performance goal. The authors
found that their adapted model significantly outperformed the classic model in all collections and
performed similarly to the probabilistic model in the German collections. Fautsch & Savoy
therefore concluded that an adapted vector-space model could be beneficial for creating effective
domain-specific information retrieval systems.

Castells et al. (2018) similarly suggest an adaptation of the vector-space model in order to
adapt to domain-specific corpora. The authors found that for ontology-based information
retrieval systems, a vector-space model that uses annotations of the documents returned better
results than a classic vector-space model system. While this study has recognized limitations,
primarily from the automatic system used to annotate the documents examined, it is an
interesting step in adapting an older method to fit modern domain needs.

Scells et al. (2018) propose not only an information retrieval system for domain-specific
documents but an entire open-source framework under which a domain-specific searching
application may be made. The proposed framework consists of four major components written in
Go: a common query representation, a parser and compiler to modify queries to fit that
representation, a pipeline for completing information retrieval experiments, and a pipeline that
can be written in domain-specific language. The experimentation pipeline is complete with
different modules that might be relevant in an information retrieval system, much like the ones

outlined in our study. This novel software, if utilized properly, can provide researchers the ability
to more easily determine what factors make an effective information retrieval system for their
domain of interest.


**3 Corpora**

***3.1 Cranfield 1400 Collection***

The Cranfield 1400 Collection is a relatively old collection of aerodynamics journal articles,
together with 226 queries, 1,400 documents, and 1,837 evaluations. The relevance evaluation
document contains query-document pair scores descending order. We used a development set of
queries and documents to develop our system and the entire collection as our test corpus.

***3.2 NFCorpus***

NFCorpus includes non-technical English queries about nutrition and academic medical paper
documents. While the 3,244 queries are topics, video descriptions, article and video titles
extracted from NutritionFacts.org, the 9,964 medical documents are mostly from PubMed. We
attempted to use the development and test sets to test our system performance on retrieving
answers that contain nutrition and medical terms.

***3.3 A Note on the BOLT Corpus***

Our original view for this project was to explore the efficacy of different modules on a large
corpus of informal language documents and queries – namely, the DARPA Broad Operational
Language Translation (BOLT) corpus (Chen et al., 2018). This corpus consists of pilot,
development, and test sets that each in turn consist of discussion threads from forums in English,
Mandarin, and Arabic and queries relevant to these threads. This corpus was developed with the
goal of combining information retrieval tasks with machine translation tasks on informal
language, such that an information retrieval system would be able to handle queries and to return
relevant documents in all three languages.

The test corpus is incredibly large, with just the English language portion containing over
three million words. The original authors, therefore, did not score the relevance of all
query-document pairs, instead only scoring those pairs retrieved by their system. For our
purposes, this meant that some pairings produced by our system might not have had relevance
assessments from the original authors, unless we were able to exactly replicate the original
system’s results, which was supremely unlikely for a couple of reasons. For one, our system
would only be considering queries and documents in English, whereas the original system used
all three languages. Even if we had used all queries and documents in all three languages, the
likelihood of four undergraduate students being able to replicate a project from the Language
Data Consortium in half a semester is completely improbable, if not actually impossible. As we
did not have enough time or resources to determine relevance scores ourselves (again, we were
working against a time limitation and with an incredibly large corpus), and we did not want to

consider hiring out the relevance assessments (as other studies using BOLT have done), we were
left a little lost. The original BOLT source code does provide a script that can be used to predict
the relevance assessment of query-document pairs that were not originally assessed. We spent
about a week figuring out how to make our system output compatible with this script before
determining that it was impractical for us to try to use for this project because of the computation
time necessary and the unreliability of the assessment scores it produced.

All in all, we spent the majority of our time on this project focusing on a corpus that we
did not end up being able to use. We originally were using the Cranfield 1400 Collection as a
way to explore the modules we wanted to use while we figured out how best to use the BOLT
corpus, and then as a formal language foil to the BOLT corpus’ informal language. We therefore
created our system and chose the modules we would use with the underlying assumption that we
would be comparing formal and informal language information retrieval. For our final tests, we
kept only those that we thought would be at all relevant to the two corpora we did end up using.
Had we used the BOLT corpus, we would have also included idiomatic dictionaries.
