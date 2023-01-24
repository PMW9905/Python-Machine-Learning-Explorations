import pandas as pd
import numpy as np
import glob
import os
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# Import local files needed
from discrete_naive_bayes import DiscreteNaiveBayes
from multi_naive_bayes import MultinomialNaiveBayes
from mcap_logistic_regression import MCAPLogisticRegression
from file_utilities import countFiles, countWordsAndFiles, recordInstances, getListOfAllFiles


def main():
    # Set directory of all files. Put in a list.
    # Also save a name list or something so I have the name output.
    # Make sure the directories are local and not otherwise.
    test_directories = ['Datasets\\enron1_test','Datasets\\enron4_test','Datasets\\hw1_test']
    train_directories = ['Datasets\\enron1_train','Datasets\\enron4_train','Datasets\\hw1_train']
    dataset_names = ['enron1','enron4','hw1']

    # For each element in list:
    for index in range(len(dataset_names)):
        print('Displaying results for '+dataset_names[index]+'\n\n')
        # Build the 4 dataframes:

        # Grab all needed dirs 
        ham_train_dir = glob.glob(train_directories[index]+"\\**\\ham", recursive=True)[0]
        ham_test_dir = glob.glob(test_directories[index]+"\\**\\ham", recursive=True)[0]
        spam_train_dir = glob.glob(train_directories[index]+"\\**\\spam", recursive=True)[0]
        spam_test_dir = glob.glob(test_directories[index]+"\\**\\spam", recursive=True)[0]

        train_dirs = [ham_train_dir]+[spam_train_dir]
        test_dirs = [ham_test_dir]+[spam_test_dir]

        # Grab vocabulary & num trainfiles
        vocabulary, train_file_count = countWordsAndFiles(train_dirs)
        test_file_count = countFiles(test_dirs)

        # First the bag models

        # Create dataframe for bag train
        bag_train_df = pd.DataFrame(columns=['row_class']+list(vocabulary),index = range(train_file_count))
        for col in bag_train_df.columns:
            bag_train_df[col].values[:] = 0
        bag_train_df[['row_class']] = bag_train_df[['row_class']].astype('string')

        # Create dataframe for bag test
        bag_test_df = pd.DataFrame(columns=['row_class']+list(vocabulary),index = range(test_file_count))
        for col in bag_test_df.columns:
            bag_test_df[col].values[:] = 0
        bag_test_df[['row_class']] = bag_test_df[['row_class']].astype('string')

        # Fill out dfs
        class_set = recordInstances(train_dirs, bag_train_df)
        recordInstances(test_dirs, bag_test_df)

        # Then construct the bernoulli from the bag models

        # Deep copy the bag dataframes and cast all vars as boolean to make the bernoulli
        bernoulli_train_df = bag_train_df.copy()
        for col in bernoulli_train_df.columns:
            if bernoulli_train_df[col].dtype != 'string':
                bernoulli_train_df[col].values[:] = bernoulli_train_df[col].values[:].astype('bool').astype('int32')
        bernoulli_test_df = bag_test_df.copy()
        for col in bernoulli_test_df.columns:
            if bernoulli_test_df[col].dtype != 'string':
                bernoulli_test_df[col].values[:] = bernoulli_test_df[col].values[:].astype('bool').astype('int32')

        # Grab list of test files & classes
        list_of_test_files, list_of_row_classes = getListOfAllFiles(test_dirs)

        # Grab 'true' string
        true_value = [value for value in class_set if 'spam' in value][0]

        # -----------------------------------------------------------------------------------------------

        # Run 1st algo and print results.

        # Train
        MultiNB = MultinomialNaiveBayes()
        MultiNB.trainModel(class_set, bag_train_df, vocabulary)

        # Test model
        result_matrix = MultiNB.testModelForAllFiles(class_set,list_of_test_files,list_of_row_classes,vocabulary,true_value)
        printStatsitics('Multinomial Naive Bayes',result_matrix)

        # -----------------------------------------------------------------------------------------------

        # Run 2nd algo and print results.

        # Train
        MultiNB = DiscreteNaiveBayes()
        MultiNB.trainModel(class_set, bernoulli_train_df, vocabulary)

        # Test model
        result_matrix = MultiNB.testModelForAllFiles(class_set,list_of_test_files,list_of_row_classes,vocabulary,true_value)
        printStatsitics('Discrete Naive Bayes', result_matrix)

        # -----------------------------------------------------------------------------------------------

        # Run 3rd algo and print results.

        # For bag

        # Split
        df_x_train, df_y_train =  splitTrainTest(bag_train_df, true_value)
        df_x_test, df_y_test = splitTrainTest(bag_test_df, true_value)

        # Find best lambda
        best_lambda = MCAPLogisticRegression.findOptimalLambda(bag_train_df)

        # Test
        logReg = MCAPLogisticRegression()
        logReg.train(df_x_train,df_y_train,best_lambda,200)

        # Train 
        logReg_results = logReg.test(df_x_test)
        result_matrix = logReg.buildResultsMatrix(logReg_results,df_y_test)
        printStatsitics('Bag of Words MCAP Logistic Regression', result_matrix)

        # -----------------------------------------------------------------------------------------------

        # For bernoulli

        # Split
        df_x_train, df_y_train =  splitTrainTest(bernoulli_train_df, true_value)
        df_x_test, df_y_test = splitTrainTest(bernoulli_test_df, true_value)

        # Find best lambda
        best_lambda = MCAPLogisticRegression.findOptimalLambda(bernoulli_train_df)

        # Test
        logReg = MCAPLogisticRegression()
        logReg.train(df_x_train,df_y_train,best_lambda,200)

        # Train 
        logReg_results = logReg.test(df_x_test)
        result_matrix = logReg.buildResultsMatrix(logReg_results,df_y_test)
        printStatsitics('Bernoulli MCAP Logistic Regression', result_matrix)
        
        # -----------------------------------------------------------------------------------------------

        # Run 4th algo and print results.
        possible_params = {
        "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
        "alpha" : [0.0001, 0.001, 0.01, 0.1],
        "penalty" : ["l2", "l1", "none"],
        }
        # For bag

        # Split
        df_x_train, df_y_train =  splitTrainTest(bag_train_df, true_value)
        df_x_test, df_y_test = splitTrainTest(bag_test_df, true_value)

        # Find best parameters & Train
        sgdClass = SGDClassifier(max_iter=100)
        sgdTuned = GridSearchCV(sgdClass, param_grid=possible_params)
        sgdTuned.fit(df_x_train,df_y_train.values.ravel())

        # Test
        results = sgdTuned.predict(df_x_test)
        result_matrix = createSGDResultsMatrix(results,df_y_test)
        printStatsitics('Bag of Words SGDClassifier', result_matrix)

        # -----------------------------------------------------------------------------------------------

        # for bernoulli

        # Split
        df_x_train, df_y_train =  splitTrainTest(bernoulli_train_df, true_value)
        df_x_test, df_y_test = splitTrainTest(bernoulli_test_df, true_value)

        # Find best parameters & Train
        sgdClass = SGDClassifier(max_iter=100)
        sgdTuned = GridSearchCV(sgdClass, param_grid=possible_params)
        sgdTuned.fit(df_x_train,df_y_train.values.ravel())

        # Test
        results = sgdTuned.predict(df_x_test)
        result_matrix = createSGDResultsMatrix(results,df_y_test)
        printStatsitics('Bernoulli SGDClassifier', result_matrix)

def createSGDResultsMatrix(results,df_y_test):
    np_y = df_y_test.to_numpy().flatten()
    result_matrix = {'TP':0,'FP':0,'TN':0,'FN':0}
    for i in range(len(results)):
        if results[i] == True:
            if np_y[i] == True:
                result_matrix['TP']+=1
            else:
                result_matrix['FP']+=1
        else:
            if np_y[i] == False:
                result_matrix['TN']+=1
            else:
                result_matrix['FN']+=1
    return result_matrix 


def printStatsitics(model_name, result_matrix):
    accuracy = (result_matrix['TP']+result_matrix['TN']) / sum(result_matrix.values())
    precision = result_matrix['TP'] / (result_matrix['TP']+result_matrix['FP'])
    recall = result_matrix['TP'] / (result_matrix['TP']+result_matrix['FN'])
    f_one = 2 * (accuracy * precision) / (recall + precision)
    print(model_name +' Results:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f_one}\n')

def splitTrainTest(df,true_value):
    df_y = df.iloc[:,0].to_frame()
    df_x = df.iloc[:,1:]
    df_y['row_class'] = np.where(df_y['row_class'] == true_value, True, False)
    return df_x, df_y


if __name__ == "__main__":
    main()



    