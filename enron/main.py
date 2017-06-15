#!/usr/bin/python

''' Completed on 17/05/17 00:35 hrs'''

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus','fraction_to_poi_email' , 'shared_receipt_with_poi','fraction_from_poi_email'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#print data_dict

features = ["salary","bonus"]
data_dict.pop('TOTAL')
data = featureFormat(data_dict,features)

# Removing "NaN"
outlier = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == "NaN":
        continue
    outlier.append((key,int(val)))

outlier_final = (sorted(outlier,key=lambda x:x[1],reverse=True)[:4])

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary,bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()

def dict_form(key,normal):
    new_list = []

    for i in data_dict:
        if data_dict[i][key] == "NaN" or data_dict[i][normal] == "NaN":
            new_list.append(0.)
        elif data_dict[i][key] >= 0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normal]))
    return new_list

fraction_from_poi = dict_form("from_poi_to_this_person" , "to_messages")
fraction_to_poi = dict_form("from_this_person_to_poi","from_messages")
#print fraction_to_poi,"\n",fraction_from_poi
cnt = 0

for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi[cnt]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi[cnt]
    cnt += 1

final_dataset = data_dict

data = featureFormat(final_dataset,features_list)

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter(from_poi,to_poi)
    if point[0] == 1: plt.scatter(from_poi,to_poi,color = "r",marker="*")

plt.xlabel("Fraction of emails this person gets from poi")
#plt.show()


labels,features = targetFeatureSplit(data)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import cross_validation

features_train , features_test , labels_train , labels_test = cross_validation.train_test_split(features,labels,test_size=0.1,random_state=42)

from sklearn.cross_validation import KFold
kf = KFold(len(labels),3)
for train,test in kf:
    features_train = [features[ii] for ii in train]
    features_test = [features[ii] for ii in test]
    labels_train = [labels[ii] for ii in train]
    labels_test = [labels[ii] for ii in test]


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

print "Decision Tree Accuracy:- " , clf.score(features_test,labels_test)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

print "KNeighbors Accuracy:- " , clf.score(features_test,labels_test)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

print " Gaussian NB Accuracy:- " , clf.score(features_test,labels_test)


dump_classifier_and_data(clf, final_dataset, features_list)