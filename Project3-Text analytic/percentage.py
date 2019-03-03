import numpy as np
import os
import pandas as pa
import nltk
from nltk.stem import WordNetLemmatizer
import re
import csv
from sklearn.model_selection import train_test_split


#import training label
#set up dictionary
def fileload(filename = 'a.csv'):
    with open(filename) as csvfile:
         data = csv.reader(csvfile)
         headers = next(data)
         dataset = []
         for line in data:
             dataset.append(line)
         csvfile.close()
    return dataset


percentage = fileload(filename='D:\\data science\\TxtProcess\\percentage.csv')
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
for filename in filenames:
    filepath = meragefiledir+'\\'
    filepath = filepath+filename
    filepaths.append(filename)
    content = read_file_as_str(filepath)
    filecontent.append(content)

#reprocessing
def reprocess(contents):
    # Find all matches
    pattern1 = re.compile(r'(?:((?:-?\d+)(?:\.\d+)?(?:%| percent)(?:age points)?))')
    pattern2 = re.compile(r'\w+(?:\-\w+)? (?:percent(?:age points)?)')
    filtered_words1 = pattern1.findall(contents)
    filtered_words2 = pattern2.findall(contents)
    filtered_word = filtered_words1 + filtered_words2
    filtered_words = []
    [filtered_words.append(i) for i in filtered_word if i not in filtered_words]
    filtered_words = list(filter(None, filtered_words))
    return filtered_words

filter_percentage = []
for content in filecontent:
    filter_word = reprocess(content)
    filter_percentage.extend(filter_word)
pd = pa.DataFrame(filter_percentage, columns=['percentage'])
pd.to_csv(r'./percentage_predict.csv', columns=['percentage'], index=False, sep=',')




