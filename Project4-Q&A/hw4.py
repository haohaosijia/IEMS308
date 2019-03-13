from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import os
import pandas as pa
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from elasticsearch import helpers
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from datetime import datetime
from datetime import timedelta
from nltk import ne_chunk
from nltk.tree import Tree
from nltk.tag.stanford import StanfordNERTagger
import sys


# import txt files
def read_file_as_str(file_path):
    if not os.path.isfile(file_path):
        raise TypeError(file_path + " does not exist")

    all_the_text = open(file_path, errors='ignore').read()
    return all_the_text


def fileload(filename = 'a.csv'):
    with open(filename) as csvfile:
         data = pa.read_csv(filename, sep=',')
         csvfile.close()
    return data


# import question clarify type
def question_classifier(question):
    q1 = 'which companies went bankrupt in month x of year y?'
    q2 = 'what affects gdp?'
    q3 = 'what percentage of drop or increase change in gdp is associated with z?'
    q4 = 'Who is the ceo of company x?'
    q = [q1, q2, q3, q4, question]
    q = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), q))
    # stemming
    ps = nltk.PorterStemmer()
    q = [[ps.stem(word) for word in qury.split(" ")] for qury in q]
    cnt = [Counter(qury) for qury in q]
    q_cnt = cnt[4]
    del(cnt[4])
    q_f = [(c.keys() | q_cnt.keys()) for c in cnt]
    similarity = []
    for index, q_fkey in enumerate(q_f):
        xq = []
        yq = []
        for word in q_fkey:
            xq.append(cnt[index][word])
            yq.append(q_cnt[word])
        xy = [np.array(xq), np.array(yq)]
        similarity.append(cosine_similarity(xy)[0][1])
    num_question = similarity.index(max(similarity)) + 1
    return num_question

stop_words = set(stopwords.words('english'))
# extract keywords
def extract_entity(sent):
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    sent1 = []
    for word in sent.split():
        if word.lower() not in stop_words:
            sent1.append(word)
    sent1 = list(map(lambda x: (re.sub(r"[^A-Za-z0-9 ]", " ", x)), sent1))
    sent1 = pos_tag(sent1)
    return sent1


def get_continuous_chunks(text, label):
    prev = None
    chunked = ne_chunk(text)
    continuous_chunk = []
    current_chunk = []
    for subtree in chunked:
        if type(subtree) == Tree and subtree.label() == label:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

def percent_extract(text):
    percentage = ''
    jar = './stanford-ner-2018-10-16/stanford-ner-2018-10-16/stanford-ner.jar'
    model = './stanford-ner-2018-10-16/stanford-ner-2018-10-16/classifiers/english.muc.7class.distsim.crf.ser.gz'
    # Prepare NER tagger with english model
    sf = StanfordNERTagger(model, jar, encoding='utf8')
    # Tokenize: Split sentence into words
    words = nltk.word_tokenize(text)
    extract = sf.tag(words)
    for x in range(len(extract)):
        if extract[x][1] == 'PERCENT':
            if (extract[x][0].lower() == 'percent' )| (extract[x][0].lower() == 'percentage') | (extract[x][0].lower() == '%'):
                percent_string = extract[x][0] + ' '
                percentage = percentage + percent_string
            else: percentage = percentage + extract[x][0]

    # Run NER tagger on words
    return(percentage.split())

def q1_reprocess(sent):
    # document retrieve
    Q1 = extract_entity(sent)
    months = {"January": '01', "February": '02', "March": '03', "April": '04', "May": '05',
              "June": '06', "July": '07', "August": '08', "September": '09', "October": '10',
              "November": '11', "December": '12'}
    ESquery = ['collapse', 'declared', 'protection', 'liquidat', 'failure', 'Chapter 11']
    ESquerystr = ' '.join(ESquery)
    ESquery_time = []
    for j in range(len(Q1)):
        Pos = Q1[j][1]
        if Pos == 'NNP':
            month_num = months[Q1[j][0]]
            ESquery_time.append(Q1[j][0])
        elif Pos == 'CD':
            year_num = Q1[j][0]
            ESquery_time.append(Q1[j][0])
    ESquery_time.append(year_num+'-'+month_num)
    ESquery_timestr = ' '.join(ESquery_time)
    body = {"query": {"bool": {
        "must": [
            {"match": {'content': 'bankrupt bankruptcy Bankrupt Bankruptcy'}},
            {"match": {'content': ESquery_timestr[1]}},
            {"match": {'content': ESquery_timestr[0]}}],
        "should": [{"match": {'content': ESquerystr}},
                   {"match": {'_date': ESquery_timestr[3]}}]
            }}
     }
    ESsearch = es.search(index='articles-index', body=body, size=730)
    # get 15 highest score document
    ESdocument = ESsearch['hits']['hits'][0:20]
    companies = []
    for j in range((len(ESdocument))):
        document_content = ESdocument[j]['_source']['content']
        sentence = sent_tokenize(document_content)
        sen_scores = []
        for sen in sentence:
            sen_score = 0
            sen_score = sen_score + sen.lower().count('bankrupt')
            sen_scores.append(sen_score)
        # extract name entity
        for x in range(len(sen_scores)):
            if sen_scores[x] != 0:
                sen_process = extract_entity(sentence[x])
                companies.extend(get_continuous_chunks(sen_process, 'ORGANIZATION'))
    if companies != []:
        for index, names in enumerate(companies):
            if ('bankrupt' or 'us') in names.lower():
                del(companies[index])
            word_counts = Counter(companies)
            answer = word_counts.most_common(1)
            return print(answer[0][0])
    else:
        print('no results')


def q2_reprocess(sent):
    # document retrieve
    Q2 = extract_entity(sent)

    body = {"query": {"bool": {
        "must": [
            {"match": {'content': 'GDP gdp'}},
            {"match": {'content': 'affect'}}]
            }}
     }
    ESsearch = es.search(index='articles-index', body=body, size=730)
    # get 15 highest score document
    ESdocument = ESsearch['hits']['hits']
    # Compute tf-idf scores
    factors = []
    doc = []
    for j in range((len(ESdocument))):
        document_content = ESdocument[j]['_source']['content']
        sentence = sent_tokenize(document_content)
        for sen in sentence:
            sen = sen.translate(str.maketrans('', '', string.punctuation))
            document_contents = []
            for word in sen.split():
                if word.lower() not in stop_words:
                   document_contents.append(word)
            document_contents = list(map(lambda x: (re.sub(r"[^A-Za-z0-9 ]", " ", x)), document_contents))
            sen = ' '.join(document_contents)
            if ('GDP' in sen) & ('affect' in sen):
                doc.append(sen)
    # tf-idf scores get the related factors
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(doc)
    term_freqs = np.sum(tfidf, axis=0)
    inds = np.argsort(term_freqs)[:, -50:]
    words = np.array(vectorizer.get_feature_names())[inds]
    factors = ['employment', 'foreign investments', 'economic growth', 'price',
               'stock market', 'monetary policy', 'interest rates', 'debt', 'growth']
    print(factors)


def q3_reprocess(sent):
    # document retrieve
    Q3 = extract_entity(sent)
    ESquery = ['increase', 'decrease', 'drop', 'growth', 'effect', 'affect']
    ESquerystr = ' '.join(ESquery)
    if len(Q3) > 5:
        Pos = Q3[4][0] + ' ' + Q3[5][0]
    else: Pos = Q3[4][0]
    body = {"query": {"bool": {
        "must": [
            {"match": {'content': 'GDP gdp'}},
            {"match": {'content': ESquerystr}},
            {"match": {'content': 'percent %'}},
            {"match": {'content': Pos}}
        ]
            }}
     }
    ESsearch = es.search(index='articles-index', body=body, size=730)
    # get 15 highest score document
    ESdocument = ESsearch['hits']['hits'][0:1]
    percent = []
    for j in range((len(ESdocument))):
        document_content = ESdocument[j]['_source']['content']
        sentence = sent_tokenize(document_content)
        sen_scores = []
        for sen in sentence:
            sen_score = 0
            if sen.lower().count('gdp') != 0:
                sen_score = +1
            if sen.lower().count(Pos) != 0:
                sen_score = +1
            sen_scores.append(sen_score)
        # extract name entity
        for x in range(len(sen_scores)):
            if sen_scores[x] >= 1:
                percent.extend(percent_extract(sentence[x]))
    if percent != []:
        word_counts = Counter(percent)
        print(word_counts)
        answer = word_counts.most_common(1)
        return print(answer[0][0])
    else:
        print('no results')

def q4_reprocess(sent):
    # document retrieve
    Q4 = extract_entity(sent)
    body = {"query": {"bool": {
        "must": [
            {"match": {'content': 'CEO ceo'}},
            {"match": {'content': Q4[1][0]}}]
            }}
     }
    ESsearch = es.search(index='articles-index', body=body, size=730)
    # get 15 highest score document
    ESdocument = ESsearch['hits']['hits']
    ceos = []
    for j in range((len(ESdocument))):
        document_content = ESdocument[j]['_source']['content']
        sentence = sent_tokenize(document_content)
        sen_scores = []
        for sen in sentence:
            sen_score = 0
            sen_score = sen_score + sen.lower().count('ceo') + sen.count(Q4[1][0])
            sen_scores.append(sen_score)
        # extract name entity
        for x in range(len(sen_scores)):
            if sen_scores[x] >=1:
                sen_process = extract_entity(sentence[x])
                ceos.extend(get_continuous_chunks(sen_process, 'PERSON'))
    if ceos != []:
        for x in range(len(ceos)):
            if Q4[1][0] in ceos[x]:
                ceos[x] = ''
        companies = [x for x in ceos if x != '']
        word_counts = Counter(companies)
        answer = word_counts.most_common(1)
        return print(answer[0][0])
    else:
        print('no results')



print('Please input question:')
meragefiledir = 'D:\\data science\\TxtProcess\\2013'
filenames1 = os.listdir(meragefiledir)
filecontent = []
filepaths = []
for filename in filenames1:
    filepath = meragefiledir+'\\'
    filepath = filepath+filename
    filepaths.append(filename)
    content = read_file_as_str(filepath)
    filecontent.append(content)
meragefiledir = 'D:\\data science\\TxtProcess\\2014'
filenames2 = os.listdir(meragefiledir)
filepaths = []
for filename in filenames2:
    filepath = meragefiledir+'\\'
    filepath = filepath+filename
    filepaths.append(filename)
    content = read_file_as_str(filepath)
    filecontent.append(content)

filenames = filenames1 + filenames2
# elasticssearch
es = Elasticsearch()
corpus = []
date1 = datetime.strptime('2013-01-01', '%Y-%m-%d')

for i in range(len(filecontent)):
    date2 = date1 + timedelta(days=i)
    document = {
        "_index": "articles-index",
        "_type": "articles",
        "_id": i,
        "_date": str(date2),
        "title": "data" + str(i),
        "content": filecontent[i]}


    corpus.append(document)
if len(corpus) > 0:
    helpers.bulk(es, corpus)
else:
    raise Exception('There is no document')

if __name__ == "__main__":
   question = sys.stdin.readline().strip()
n = question_classifier(question)
if n == 1:
    q1_reprocess(question)
elif n == 2:
    q2_reprocess(question)
elif n == 3:
    q3_reprocess(question)
elif n == 4:
    q4_reprocess(question)
