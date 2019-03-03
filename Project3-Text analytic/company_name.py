import numpy as np
import os
import pandas as pa
import nltk
from nltk.stem import WordNetLemmatizer
import re
import csv
from sklearn.model_selection import train_test_split


#import training label
def fileload(filename = 'a.csv'):
    with open(filename) as csvfile:
         data = csv.reader(csvfile)
         headers = next(data)
         dataset = []
         for row in data:
             dataset.append(row[0])
         csvfile.close()
    return dataset
companies = fileload(filename='D:\\data science\\TxtProcess\\companies.csv')
# import txt files
def read_file_as_str(file_path):
    if not os.path.isfile(file_path):
        raise TypeError(file_path + " does not exist")

    all_the_text = open(file_path, errors='ignore').read()
    return all_the_text

meragefiledir = 'D:\\data science\\TxtProcess\\2013'
filenames = os.listdir(meragefiledir)
filecontent = []
filepaths = []
for filename in filenames:
    filepath = meragefiledir+'\\'
    filepath = filepath+filename
    filepaths.append(filename)
    content = read_file_as_str(filepath)
    filecontent.append(content)
meragefiledir = 'D:\\data science\\TxtProcess\\2014'
filenames = os.listdir(meragefiledir)
filepaths = []
for filename in filenames:
    filepath = meragefiledir+'\\'
    filepath = filepath+filename
    filepaths.append(filename)
    content = read_file_as_str(filepath)
    filecontent.append(content)

#reprocessing
def reprocess(contents):
    # Find all matches
    pattern = re.compile(r'(?:[A-Z][A-Za-z\-]+ ?)+')
    filtered_word = pattern.findall(contents)
    filtered_w = []
    for w in filtered_word:
        if w[len(w)-1] == ' ':
            w = w[:-1]
        filtered_w.append(w)
    filtered_words = []
    [filtered_words.append(n) for n in filtered_w if n not in filtered_words]
    proper_name = ['Northern', 'San Francisco', 'Mexico', 'Fiscal', 'Rate', 'Six', 'Seven', 'Eight', 'Republican',
                   'Latin', 'Kingdom', 'Senate', 'Report', 'Dakota', 'Average', 'Europe', 'Speaker',
                   'Angeles', 'Russia', 'Middle', 'Tax', 'Act', 'DC', 'UK', 'Officer', 'Nine', 'Good',
                   'President', 'Coast', 'Credit', 'Blog', 'Security', 'District', 'Home', 'Chair', 'Court',
                   'Incomes', 'Trade', 'Tuesday', 'Wednesday', 'Texa', 'Morning', 'Eye', 'Democratic''Thursday',
                   'Friday', 'Saturday', 'Sunday', 'Year', 'Yesterday', 'Democrat', 'Growth', 'League', 'Money',
                   'Council', 'Leader', 'Majority', 'Last', 'Minority', 'Economy', 'Stocks', 'Dream', 'Private',
                   'University', 'Department', 'Sport', 'Class', 'Privacy', 'Electricity', 'Asia', 'College', 'Index']
    # normalization
    lemmatizer = WordNetLemmatizer()
    filtered_words = list(map(lambda x: lemmatizer.lemmatize(x), filtered_words))
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    for index, name in enumerate(filtered_words):
        spl_name = name.split()
        for word in stop_words:
            for sp in spl_name:
                if word == 'the':
                    break
                if word.capitalize() == sp:
                    filtered_words[index] = ''
    filtered_words = list(filter(None, filtered_words))
    for index, name in enumerate(filtered_words):
        for proper in proper_name:
            for sp in spl_name:
                if proper == sp:
                    filtered_words[index] = ''
    filtered_words = list(filter(None, filtered_words))
    return filtered_words


# feature set up
def feature(contents, names):
    # sentence and word tokenize
    def tokenize(con):
        sentence = []
        sens = nltk.sent_tokenize(con)
        for x in sens:
            sentence.append(x)
        wo = []
        for index in sentence:
            w = nltk.word_tokenize(index)
            for t in w:
                wo.append(t)
        return wo, sentence
    fact = []
    len_first = []
    len_last = []
    ceo_sentence_count = []
    financ_sentence = []
    capital_sentence = []
    com_sentence = []
    len_sentence = []
    names_in_contents = []
    word_in_name = []
    index_firsts = []
    com_content = []
    capital_letter = []
    number_sentence = []
    len_name_word = []
    len_name_char = []
    word = ['Inc', 'Co', 'Corp', 'Corporation', 'Ltd', 'Capital', 'Company', 'Financial',
            'Group', 'Management']
    [words, sentences] = tokenize(contents)
    for name in names:
        # look up companies decide whether it is a companies -- fact
        if name in companies:
            fact.append(1)
        else: fact.append(0)
        # frequency of names in content
        names_in_contents.append(contents.count(name))
        str_name = name.split(" ")
        # length of first name and last name
        len_first.append(len(str_name[0]))
        if len(str_name) == 1:
            len_last.append(0)
        else:
            len_last.append(len(str_name[len(str_name)-1]))
        len_name_word.append(len(name.split()))
        len_name_char.append(sum(len(s) for s in str_name))
        name_sen = ''
        # find the sentence that have the name, calculate number
        count_sentences = 0
        for sen in sentences:
            if name in sen:
                name_sen = name_sen + ' ' + sen
                count_sentences = count_sentences + 1
        number_sentence.append(count_sentences)
        len_sentence.append(len(name_sen))
        count_capitalize = 0
        for st in str_name:
            # capital letter in name
            for s in st:
                if s.isupper():
                    count_capitalize = count_capitalize + 1
        capital_letter.append(count_capitalize)
        for st in str_name:
            # word in names
            if st in word:
                w_in_name = 1
                break
            else: w_in_name = 0
        word_in_name.append(w_in_name)
        # frequency of ceo and Chief executive officer in sentence
        ceo_sentence_count.append(name_sen.lower().count('ceo')
                                  + name_sen.lower().count('chief')
                                  + name_sen.lower().count('executive')
                                  + name_sen.lower().count('officer'))
        # find the words count in sentence
        words = nltk.word_tokenize(name_sen)
        # Word index of first word
        try:
            index_first = words.index(str_name[0])
        except ValueError:
            index_first = 0
        index_firsts.append(index_first)
        # frequency of company
        com_sentence.append(name_sen.lower().count('company'))
        com_content.append(contents.lower().count('company'))
        financ_sentence.append(name_sen.lower().count('financ'))
        capital_sentence.append(name_sen.lower().count('capital'))
    # set up feature DataFrame
    features = pa.DataFrame({"fact": fact})
    features["name"] = names
    features["names_in_contents"] = names_in_contents
    features["com_content"] = com_content
    features["com_sentence"] = com_sentence
    features["ceo_in_contents"] = contents.lower().count('ceo')
    features["ceo_sentence_count"] = ceo_sentence_count
    features["financ_content"] = contents.lower().count('financ')
    features["financ_sentence"] = financ_sentence
    features["capital_content"] = contents.lower().count('capital')
    features["capital_sentence"] = capital_sentence
    features["word_in_name"] = word_in_name
    features["number_sentence"] = number_sentence
    features["capital_letter"] = capital_letter
    features["len_name_word"] = len_name_word
    features["len_name_char"] = len_name_char
    # length of article
    features["len_first"] = len_first
    features["len_last"] = len_last
    features["len_content"] = len(contents)
    features["len_sentence"] = len_sentence
    features['index_first'] = index_firsts
    return features


# set train set and test set
all_feature = pa.DataFrame()
for i, content in enumerate(filecontent):
    filter_word = reprocess(content)
    feature_get = feature(content, filter_word)
    all_feature = pa.concat([all_feature, feature_get])
    if i % 10 == 0:
        print(f'The program has already processed {i} articles')
feature, label = all_feature.iloc[:, 1:22], all_feature.iloc[:, [0]]
count_features = all_feature['fact'].value_counts()
negative_count = count_features[0]
print(negative_count)
postive_count = count_features[1]
print(postive_count)
# tran set -- 70% test set -- 30%
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3, random_state=0)
# nerual model set up
from sklearn.neural_network import MLPClassifier
feature_train1 = feature_train.iloc[:, 1:21].values
label_train1 = label_train.values
feature_test1 = feature_test.iloc[:, 1:21].values
label_test1 = label_test.values.ravel()
feature1 = feature.iloc[:, 1:21].values
# scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# fit only on training data
scaler.fit(feature_train1)
feature_train1 = scaler.transform(feature_train1)
# apply same transformation to test data
feature_test1 = scaler.transform(feature_test1)
feature1 = scaler.transform(feature1)
# over sample to balance training set
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
feature_train_resampled, label_train_resampled = ros.fit_sample(feature_train1, label_train1)
clf = MLPClassifier()
clf.fit(feature_train_resampled, label_train_resampled)
predict2 = clf.predict(feature1)
predict1 = clf.predict(feature_test1)
candidate = feature_test.iloc[:, 0].values


def print_metrics(truth, predicted):
    # confusion matrix
    from sklearn import metrics
    cm = metrics.confusion_matrix(truth, predicted)
    acc = metrics.accuracy_score(truth, predicted)
    prec = metrics.precision_score(truth, predicted)
    rec = metrics.recall_score(truth, predicted)
    f1 = metrics.f1_score(truth, predicted)
    print(f"Confusion Matrix:\n {cm}\n")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    return
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(all_feature.iloc[:, 2:22].values, all_feature.iloc[:, 0].values, discrete_features=True)
my_matrix1 = np.c_[candidate, predict1, label_test1, feature_test.iloc[:, 1:21].values]
pd1 = pa.DataFrame(my_matrix1)
pd1.to_csv(r'./company_feature.csv', index=False, sep=',')
my_matrix2 = np.c_[feature.iloc[:, 0], predict2, label.values]
pd2 = pa.DataFrame(my_matrix2)
pd2.to_csv(r'./company_predict.csv', index=False, sep=',')
print_metrics(label_test1, predict1)
print_metrics(label_train1, predict2)
print(mi)