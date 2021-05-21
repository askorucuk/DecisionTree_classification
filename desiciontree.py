# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:21:07 2021

@author: askor
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



wine = load_wine()

print('Classes : ', wine.target_names , "\n");
print('Features : ', wine.feature_names , "\n");

X = wine.data
y = wine.target

clf = DecisionTreeClassifier(random_state=0);

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state = 0, 
                                                    test_size = 0.25, stratify = y);

clf.fit(X_train, y_train);

result_data = clf.predict(X_test);

print("Result : ",result_data ,"\n");

print("\nAccuracy score = " + str(accuracy_score(result_data, y_test))+"\n");

feature_imp = pd.Series(clf.feature_importances_,index=wine.feature_names).sort_values(ascending=False)
print(feature_imp);

cm = confusion_matrix(y_test, result_data);
print("\nConfusion Matrix : ");
print(cm,"\n");

print("\nCross Validation Score : " , cross_val_score(clf,wine.data,wine.target,cv=10));

n = float(input("Please enter the number of features : "));
test_arr1 = input("Please enter the your test data(separate your test data with commas) : ");
test_list1 = list(map(float,test_arr1.split(',')));
n = float(input("Please enter the number of features : "));
test_arr2 = input("Please enter the your second test data(separate your test data with commas) : ");
test_list2 = list(map(float,test_arr2.split(',')));
main_test_list = (test_list1,test_list2);

input_test_result = clf.predict(main_test_list);
print("\nUser Input Test Result :",input_test_result);
