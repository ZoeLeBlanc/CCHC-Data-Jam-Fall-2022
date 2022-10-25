import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np

def get_spacy_subjects(df):
    unique_subjects = pd.DataFrame([{'unique_subject': df.subject.unique().tolist(), 'spacy_label': None}]).explode('unique_subject')
    import spacy
    nlp = spacy.load('en_core_web_trf')
    def get_spacy_label(row):
        doc = nlp(row.unique_subject)
        label = doc.ents[0].label_ if len(doc.ents) > 0 else row.spacy_label
        return label
    tqdm.pandas(desc='Getting NER labels')
    unique_subjects['spacy_label'] = unique_subjects.progress_apply(get_spacy_label, axis=1)
    return unique_subjects

def get_tfidf_vectorizer(df, column_name, max_features=1000):

    all_docs = df[column_name].tolist()

    vectorizer = TfidfVectorizer(max_df=.7, min_df=1, use_idf=True, norm=None, stop_words=stopwords.words('english'), ngram_range=(1, 2), max_features=max_features)
    transformed_documents = vectorizer.fit_transform(all_docs)

    transformed_documents_as_array = transformed_documents.toarray()

    all_files = df.id.tolist()
    tfidf_results = []
    for counter, doc in enumerate(tqdm(transformed_documents_as_array, total=len(transformed_documents_as_array), desc='Getting TFIDF Scores')):
        # construct a dataframe
        tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
        one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)
        one_doc_as_df['id'] = all_files[counter]
        tfidf_results.append(one_doc_as_df)
    tfidf_df = pd.concat(tfidf_results)
    return tfidf_df


# contributor_tfidf_df = tfidf_model(merged_df[(merged_df.contributor_exists == True) & (merged_df.cleaned_date < '1880-01-01')])
# contributor_tfidf_df[0:100]
# merged_df[['cleaned_description', 'cleaned_date']][0:100].to_csv('test_word_stream.csv', index=False)

def get_count_vectorizer(df, column_name, max_features=1000):
    count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 1), max_features=max_features)
    X = count_vectorizer.fit_transform(df[column_name].tolist()) 
    count_vect_df = pd.DataFrame(np.asarray(X.sum(axis=0)), columns=count_vectorizer.get_feature_names_out())
    description_vocabulary = count_vect_df.T
    description_vocabulary = description_vocabulary.reset_index()
    description_vocabulary = description_vocabulary.rename(columns={'index': 'word', 0: 'count'})
    description_vocabulary = description_vocabulary.sort_values('count', ascending=False)
    return description_vocabulary