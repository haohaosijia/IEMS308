import numpy as np
import os
import pandas as pa
import nltk
from nltk.stem import WordNetLemmatizer
import re
import csv
import string
from sklearn.model_selection import train_test_split


#import training label
#set up dictionary
def fileload(filename = 'a.csv'):
    with open(filename) as csvfile:
         data = csv.reader(csvfile)
         headers = next(data)
         dataset = []
         for row in data:
             if row[1] == '':
                line = row[0]
             else:
                line = row[0] + ' ' + row[1]
             dataset.append(line)
         csvfile.close()
    return dataset
ceo = fileload(filename='D:\\data science\\TxtProcess\\ceo.csv')
#companies = fileload(filename='D:\\data science\\TxtProcess\\companies.csv')

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
    exp1 = "(?=([A-Z][a-z]+ [A-Z][a-z]+))"
    exp2 = "(?=([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+))"
    pattern1 = re.compile(exp1)
    pattern2 = re.compile(exp2)
    filtered_words1 = pattern1.findall(contents)
    filtered_words2 = pattern2.findall(contents)
    filtered_word = filtered_words1 + filtered_words2
    filtered_words = []
    proper_name = ['South', 'East', 'San Francisco', 'Mexico', 'Hong Kong', 'House', 'Fiscal', 'Venture', 'Rate',
                   'Latin', 'York', 'Kingdom', 'Africa', 'Europe', 'China', 'Western', 'Northern', 'Senate', 'Report',
                   'India', 'Korea', 'Chinese', 'Morgan', 'Google', 'State', 'Dakota', 'Images', 'Average',
                   'Angeles', 'Market', 'Russia', 'Finance', 'Vegas', 'Middle', 'Estate', 'Bank', 'Tax', 'Act',
                   'DC', 'UK', 'World', 'Zealand', 'Officer', 'Times', 'United', 'Association', 'Nation', 'News'
                   'President', 'Coast', 'Busi', 'Street', 'USA', 'Credit', 'Economics', 'Blog', 'Security',
                   'Today', 'District', 'Home', 'Chair', 'Court', 'Wealth', 'Service', 'Russia', 'Speaker', 'Network',
                   'Century', 'Income', 'Trade', 'Tuesday', 'Wednesday', 'Texa', 'Global', 'Morning', 'Eye', 'Social',
                   'Thursday', 'Friday', 'Saturday', 'Sunday', 'Year', 'Yesterday', 'Policy', 'Democrat', 'Growth',
                   'League', 'Money', 'Council', 'Leader', 'Government', 'Majority', 'Industry', 'Last', 'Innovation',
                   'Minority', 'Economy', 'Economist', 'Stocks', 'Public', 'Retail', 'Street', 'Tech', 'Dream',
                   'Private', 'Boston', 'Corp', 'Univers', 'System', 'Department', 'Avenue', 'Union', 'Sport',
                   'Class', 'Wall', 'Advance', 'Priva', 'Elec', 'Asia', 'Pac', 'College', 'Index', 'Research']
    [filtered_words.append(i) for i in filtered_word if  i not in filtered_words]
    # normalization
    lemmatizer = WordNetLemmatizer()
    filtered_words = list(map(lambda x: lemmatizer.lemmatize(x), filtered_words))
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    for index, name in enumerate(filtered_words):
        spl_name = name.split()
        for word in stop_words:
            if len(spl_name) == 2:
                if word.capitalize() == spl_name[0] or word.capitalize() == spl_name[1]:
                    filtered_words[index] = ''
    filtered_words = list(filter(None, filtered_words))
    for index, name in enumerate(filtered_words):
        for proper in proper_name:
            if proper in name:
                filtered_words[index] = ''
    filtered_words = list(filter(None, filtered_words))
    return filtered_words


# feature set up
def feature(contents, names):
    # sentence and word tokenize
    def tokenize(con):
        sentence = []
        sens = nltk.sent_tokenize(con)
        for sen in sens:
            sentence.append(sen)
        words = []
        for index in sentence:
            w = nltk.word_tokenize(index)
            for i in w:
                words.append(i)
        return words,sentence
    fact = []
    len_first = []
    len_second = []
    ceo_sentence_count = []
    before_capital = []
    after_capital = []
    com_sentence = []
    len_sentence = []
    names_in_contents = []
    first_content = []
    index_firsts = []
    bus_content = []
    bus_sentence = []
    com_content = []
    [word, sentences] = tokenize(contents)
    for name in names:
        # look up ceos decide whether it is a ceo -- fact
        if name in ceo:
            fact.append(1)
        else: fact.append(0)
        # frequency of names in content
        names_in_contents.append(contents.count(name))
        str_name = name.split(" ")
        # length of first name and second name
        if len(str_name) == 3:
            len_first.append(len(str_name[0]))
            len_second.append(len(str_name[1]+str_name[2]))
        else:
            len_first.append(len(str_name[0]))
            len_second.append(len(str_name[1]))
        name_sen = ''
        # find the sentence that have the name
        for sen in sentences:
            if name in sen:
                name_sen = name_sen + ' ' + sen
        len_sentence.append(len(name_sen))
        # frequency of ceo and Chief executive officer
        ceo_sentence_count.append(name_sen.lower().count('ceo')
                                  + name_sen.lower().count('chief')
                                  + name_sen.lower().count('executive')
                                  + name_sen.lower().count('officer'))
        # find the words count in sentence
        words = nltk.word_tokenize(name_sen)
        # Word index of first word
        try:
            if len(str_name) == 3:
                index_first = words.index(str_name[0])
                index_last = words.index(str_name[2])
            else:
                index_first = words.index(str_name[0])
                index_last = words.index(str_name[1])
        except ValueError:
            index_first = 0
            index_last = len(words)
        index_firsts.append(index_first)
        # Whether the word before the first name is capitalized
        before_ind = index_first - 1
        if before_ind == -1:
            before_c = 0
        else:
            before_c = int(words[before_ind].istitle())
        # Whether the word after the last name is capitalized
        after_ind = index_last + 1
        if after_ind >= len(words):
            after_c = 0
        else:
            after_c = int(words[after_ind].istitle())
        before_capital.append(before_c)
        after_capital.append(after_c)
        first_content.append(contents.count(str_name[0]))
        com_sentence.append(name_sen.lower().count('company'))
        com_content.append(contents.lower().count('company'))
        bus_content.append(contents.lower().count('business'))
        bus_sentence.append(name_sen.lower().count('business'))
    # set up feature DataFrame
    features = pa.DataFrame({"fact": fact})
    features["name"] = names
    features["names_in_contents"] = names_in_contents
    features["ceo_in_contents"] = contents.lower().count('ceo')
    features["chief_in_contents"] = contents.lower().count('chief')
    features["executive_in_contents"] = contents.lower().count('executive')
    features["officer_in_contents"] = contents.lower().count('officer')
    features["ceo_sentence_count"] = ceo_sentence_count
    features["len_first"] = len_first
    features["len_second"] = len_second
    # length of article
    features["len_article"] = len(contents)
    features["len_sentence"] = len_sentence
    features["before_capital"] = before_capital
    features["after_capital"] = after_capital
    features["com_content"] = com_content
    features["com_sentence"] = com_sentence
    features["bus_content"] = bus_content
    features["bus_sentence"] = bus_sentence
    features['index_first'] = index_firsts
    features['first_content'] = first_content
    return features


# set train set and test set
all_feature = pa.DataFrame()
for i, content in enumerate(filecontent):
    filter_word = reprocess(content)
    feature_get = feature(content, filter_word)
    all_feature = pa.concat([all_feature, feature_get])
    if i % 10 == 0:
        print(f'The program has already processed {i} articles')

# filter_word = reprocess(filecontent[0])
# all_feature = feature(filecontent[0], filter_word)
feature, label = all_feature.iloc[:, 1:20], all_feature.iloc[:, [0]]
count_features = all_feature['fact'].value_counts()
negative_count = count_features[0]
print(negative_count)
postive_count = count_features[1]
print(postive_count)
# use cross validation
# tran set -- 70% test set -- 30%
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.3, random_state=0)

#nerual model set up
from sklearn.neural_network import MLPClassifier
feature_train1 = feature_train.iloc[:, 1:19].values
label_train1 = label_train.values
feature_test1 = feature_test.iloc[:, 1:19].values
label_test1 = label_test.values.ravel()
feature1 = feature.iloc[:, 1:19].values
label1 = label.values
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
mi = mutual_info_classif(all_feature.iloc[:, 2:20].values, all_feature.iloc[:, 0].values, discrete_features=True)
my_matrix1 = np.c_[candidate, predict1, label_test1, feature_test.iloc[:, 1:19].values]
pd1 = pa.DataFrame(my_matrix1)
pd1.to_csv(r'./ceo_feature.csv', index=False, sep=',')
my_matrix2 = np.c_[feature.iloc[:, 0], predict2, label1]
pd2 = pa.DataFrame(my_matrix2)
pd2.to_csv(r'./ceo_predict.csv', index=False, sep=',')
print_metrics(label_test1, predict1)
print_metrics(label1, predict2)
print(mi)


