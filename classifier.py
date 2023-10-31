
# Import Required Modules.
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from chefboost import Chefboost as chef

class Classifier:
    
    def __init__(self, path, csv):

        self.df = pd.read_csv(csv)
        self.x = self.df.iloc[:, path]

        # Input_ y_Target_Variable.
        self.y = self.df.iloc[:, -1]
        
        # Normalize
        scaler = MinMaxScaler()
        # scaler.fit(self.x)
        self.x_scaled = scaler.fit_transform(self.x)


    def classify(self, x_scaled, x, y, df, config):
        # Create StratifiedKFold object.
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        lst_accu_stratified = []
        accu_list = []
        precision_list = []
        recall_list = []
        auc_list = []

        for train_index, test_index in skf.split(x_scaled, y):
            x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            model = chef.fit(self.df, config = config, target_label = 'label')
            
            # classifier.fit(x_train_fold, y_train_fold)
            # lst_accu_stratified.append(classifier.score(x_test_fold, y_test_fold))

            #Revisar 
            #Hacer greatSearch
            y_pred = model.predict(x_test_fold)
            
            # Metrics
            # mse = mean_squared_error(y_test_fold, y_pred)
            # accu_list.append(accuracy_score(y_test_fold, y_pred))
            # precision_list.append(precision_score(y_test_fold, y_pred))
            # recall_list.append(recall_score(y_test_fold, y_pred))
            # auc_list.append(roc_auc_score(y_test_fold, y_pred))

        # Print the output.
        # print('List of possible accuracy:', lst_accu_stratified)
        # print('\nMaximum Accuracy That can be obtained from this model is:',
        #     max(lst_accu_stratified)*100, '%')
        # print('\nMinimum Accuracy:',
        #     min(lst_accu_stratified)*100, '%')
        # print('\nOverall Accuracy:',
        #     mean(lst_accu_stratified)*100, '%')
        # print('\nStandard Deviation is:', stdev(lst_accu_stratified))

        # print('Metrics Results:')
        # print('-'*30)
        # print(f"Accuracy: {mean(accu_list)*100:.2f}% ± {stdev(accu_list):.2f}")
        # print(f"Precision: {mean(precision_list)*100:.2f}% ± {stdev(precision_list):.2f}")
        # print(f"Recall: {mean(recall_list)*100:.2f}% ± {stdev(recall_list):.2f}")
        # print(f"AUC: {mean(auc_list)*100:.2f}% ± {stdev(auc_list):.2f}\n")

        # return np.mean(accu_list)

    def run(self):

        print('##############################################################       ID3')

        configID3 = {'algorithm': 'ID3'}
        self.classify(self.x_scaled, self.x, self.y, self.df, configID3)
        
        print('##############################################################       C4.5')
        
        configC45 = {'algorithm': 'C4.5'}
        self.classify(self.x_scaled, self.x, self.y, self.df, configC45)
        
        print('##############################################################       CART')
       
        configCART = {'algorithm': 'CART'}
        self.classify(self.x_scaled, self.x, self.y, self.df, configCART)
