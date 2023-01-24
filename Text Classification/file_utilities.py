import pandas as pd
import numpy as np
import glob
import os
import re

def countWordsAndFiles(dirs):
    word_set = set()
    num_files = 0
    for dir in dirs:
        for file in os.listdir(dir):
            num_files+=1
            with open(dir+'/'+file,'r',encoding="unicode_escape") as f:
                words = re.split(' |\n',f.read())
                for word in words:
                    word_set.add(word)
    return (word_set,num_files)

def countFiles(dirs):
    num_files = 0
    for dir in dirs:
        num_files+= len(os.listdir(dir))
    return num_files

def recordInstances(dirs,df):
    start_index = 0
    class_set = set()
    for dir in dirs:
        row_class = os.path.basename(dir)
        class_set.add(row_class)
        for file in os.listdir(dir):
            with open(dir+'/'+file,'r',encoding="unicode_escape") as f:
                words = re.split(' |\n',f.read())
                df.at[start_index,'row_class'] = row_class
                for word in words:
                    if word in df.columns:
                        df.at[start_index,word]+=1
                start_index+=1
    return class_set


def getListOfAllFiles(dirs):
    list_of_files = []
    list_of_row_classes = []
    for dir in dirs:
        row_class = os.path.basename(dir)
        for file in os.listdir(dir):
            list_of_files.append(dir+'/'+file)
            list_of_row_classes.append(row_class)
    return list_of_files, list_of_row_classes

