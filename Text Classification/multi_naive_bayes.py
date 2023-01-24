import numpy as np
import pandas as pd
import math
import re

class MultinomialNaiveBayes:
    def __init__(self):
        pass

    def trainModel(self,class_set,train_df,vocabulary):
        self.condprob = pd.DataFrame(columns=list(vocabulary),index=list(class_set))
        self.num_docs = train_df.shape[0]
        self.prior = {}
        for col in self.condprob.columns:
            self.condprob[col].values[:] = 0
        for c in class_set:
            c_df = train_df.loc[train_df['row_class'] == c]
            n_c = c_df.shape[0]
            self.prior[c]=math.log(n_c+1) - math.log(self.num_docs+1)
            class_word_count = 0
            for vocab in vocabulary:
                class_word_count += sum(c_df[vocab].values[:]) + 1
            for vocab in vocabulary:
                self.condprob.at[c,vocab] = math.log(sum(c_df[vocab].values[:]) + 1) - math.log(class_word_count)

    def testModel(self, class_set, file, vocabulary):
        word_dict = self.extractTokens(file,vocabulary)
        score = {}
        for c in class_set:
            score[c] = self.prior[c]
            for word in word_dict:
                score[c] += self.condprob.at[c,word]
        return max(score, key=score.get)

    def extractTokens(self, file, vocabulary):
        word_set = set()
        with open(file,encoding="unicode_escape") as f:
            words = re.split(' |\n',f.read())
            for word in words:
                if word in vocabulary:
                    word_set.add(word)
        return word_set

    def testModelForAllFiles(self, train_classes, test_files, test_Y, vocabulary, true_value):
        total_files = len(test_files)
        result_matrix = {'TP':0,'FP':0,'TN':0,'FN':0}

        for i in range(total_files):
            result = self.testModel(train_classes, test_files[i], vocabulary)
            if result == true_value:
                if true_value == test_Y[i]:
                    result_matrix['TP']+=1
                else:
                    result_matrix['FP']+=1
            else:
                if true_value != test_Y[i]:
                    result_matrix['TN']+=1
                else:
                    result_matrix['FN']+=1

        return result_matrix