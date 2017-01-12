#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Explore the dataset
def exploration(data_dict):
    '''given a dictionary, print the number of keys, number of features, an example data point, 
    number and percentage of poi'''
    
    total_people = len(data_dict)
    print "number of people in the dataset:", total_people
    # print "the dataset information: ", data_dict
    
    total_keys = len(data_dict['METTS MARK'])
    print "poi is the label, number of all other features in the dataset is", total_keys-1
    print "an example entry METTS MARK in the dataset: "
    pprint.pprint(data_dict['METTS MARK'])
    
    ### count POI in the dataset
    total_poi = 0
    for k in data_dict:
        if data_dict[k]["poi"] == True:
            total_poi = total_poi + 1
    print "number of poi in the dataset: ", total_poi
    print "percentage of poi in the dataset: ", 1.0*total_poi/total_people

exploration(data_dict) 

### Task 2: Remove outliers
features = ['salary','bonus']
data = featureFormat(data_dict, features) ##data_format by extract data to numpy array
#visualize the data and find 4 more outliers with high income noticed
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

#outlier is the TOTAL key
#remove a key from a dict, change the dict itself
data_dict.pop( 'TOTAL', 0 ) ### the total is an aggregation of all salary and bonus

features = ['salary','bonus']
data = featureFormat(data_dict, features) ##data_format by extract data to numpy array

#visualize the data and find 4 more outliers with high income noticed
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()    

### Check if any other key can be people's names
keys = []
for k in data_dict:
    keys.append(k)

keys = sorted(keys)
print keys ### notice 'THE TRAVEL AGENCY IN THE PARK' is not a name.

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0 ) ### the total is an aggregation of all salary and bonus
### After removing the 'TOTAL' key, summarize the dataset
exploration(data_dict)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### REMOVE FEATURES
### From the pdf document, 'total_payments' and 'total_stock_value' are linear sum of other variables. I decide to remove them from the features list. 
### 'email_address' is a text string and corresponds to the person's name, I decide to remove it from the features list as well. 

### CREATE NEW FEATURES
'''
from_poi_to_this_person_ratio = from_poi_to_this_person/to_messages
from_this_person_to_poi_ratio = from_this_person_to_poi/from_messages
'''

for e in data_dict:
    if data_dict[e]['from_poi_to_this_person'] != 'NaN':
        data_dict[e]['from_poi_to_this_person_ratio'] = 1.*data_dict[e]['from_poi_to_this_person']/data_dict[e]['to_messages']
    else:
        data_dict[e]['from_poi_to_this_person_ratio'] = 'NaN'
        
for e in data_dict:        
    if data_dict[e]['from_this_person_to_poi'] != 'NaN':
        data_dict[e]['from_this_person_to_poi_ratio'] = 1.*data_dict[e]['from_this_person_to_poi']/data_dict[e]['from_messages']
    else:
        data_dict[e]['from_this_person_to_poi_ratio'] = 'NaN'

features_list  = ['poi',
                  'salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','from_poi_to_this_person_ratio','from_this_person_to_poi_ratio']



data_array = featureFormat(data_dict, features_list)
poi, features = targetFeatureSplit(data_array)

### split the data into train and test

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, poi, test_size=0.3, random_state=42)      

###Feature scaling before selection, necessary for SVM. 
scaler = MinMaxScaler()
rescaled_features_train = scaler.fit_transform(features_train)
rescaled_features_test = scaler.fit_transform(features_test)

### Univariate selection
### Using SelectKBest, I can select features according to the k highest scores.

from sklearn.feature_selection import SelectKBest 
selection = SelectKBest(k=1)

from sklearn.pipeline import Pipeline, FeatureUnion
combined_features = FeatureUnion([("univ_select", selection)])

features_transformed = selection.fit(rescaled_features_train, labels_train).transform(rescaled_features_train)

svm = SVC(kernel="linear")

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__univ_select__k=[1, 2, 4, 6],
                  svm__C=[1,10,1e2,1e3])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv = 5, verbose=10)
grid_search.fit(rescaled_features_train, labels_train) ###only use train set
print(grid_search.best_estimator_)

###SelectKBest score
selector2 = SelectKBest()
selector2.fit(rescaled_features_train, labels_train)
scores =  selector2.scores_
print scores

### distribution of score in selectkbest
plt.hist(scores,bins = 19)
plt.title("SelectKBest score disctribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()

### features of high importance
i = 0
N = len(scores)
features_score = []
while i < N:
    features_score.append((features_list[i+1],scores[i]))
    i = i + 1
features_score = sorted(features_score, key = lambda x: x[1],reverse = True)
print features_score

### Final feature selection
features_list = ['poi', 'exercised_stock_options', 'from_this_person_to_poi_ratio', 'expenses','salary']



### find NaN features for a given variable
def qualify(data,variable):
    """ in a dictionary data, for the key variable, return the number of non-NaN value. 
    """
    total_count = 0
    for k in data:
        if data[k][variable] == 'NaN':
            total_count = total_count + 1
    return total_count, 1.* total_count/len(data)

for f in features_list :
    print 'number and percentage of missing values for', f, ':', qualify(data_dict,f)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

### at this moment, the data has been updated with outlier removal and new features added
my_dataset = data_dict

### visualize poi

data = featureFormat(my_dataset, features_list, sort_keys = True)

### Any 2 features
N = len(features_list) 
features_index_list =  range(1,N)
two_features = itertools.combinations(features_index_list, 2)

for pair in two_features:
    i = pair[0]
    j = pair[1]
    for point in data:
        poi = point[0]
        if (poi - 0.0) < 0.001: # poi == 0
            point_color = 'b'
        else:
            point_color = 'r'
        f1 = point[i]
        f2 = point[j]
        plt.scatter(f1,f2,color = point_color)
    
    plt.xlabel(features_list[i])
    plt.ylabel(features_list[j])
    plt.show()

########################################################################################################################

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.

# Provided to give you a starting point. Try a variety of classifiers.

# Example starting point. Try investigating other evaluation techniques!

### first look at financial data and plot POI in red or blue


my_poi, my_features = targetFeatureSplit(data)

names = ["Naive Bayes","Linear SVM", "RBF SVM","Decision Tree"]
    
classifiers = [
        GaussianNB(),
        SVC(kernel="linear",C = 1),
        SVC(kernel="rbf",gamma=0.1,C = 1000),
        DecisionTreeClassifier()
        ]
    
features = np.array(my_features)
poi = np.array(my_poi)

def performance(clf,features,poi):
    '''
    split features, labels using StratifiedShuffleSplit and
    calculate the total performance metrics: accuracy, precision, recall, and F1-score of a classifier clf
    '''
    cv = StratifiedShuffleSplit(poi, 1000, random_state=0) #test_size = 0.1
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_index, test_index in cv:
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = poi[train_index], poi[test_index]
        scaler = MinMaxScaler()
        rescaled_features_train = scaler.fit_transform(features_train)
        rescaled_features_test = scaler.fit_transform(features_test)

        ### fit the classifier using training set, and test on test set
        clf.fit(rescaled_features_train, labels_train)
        predictions = clf.predict(rescaled_features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

for name, clf in zip(names, classifiers):
    print '################'
    print 
    print  name
    print
    print '################'
    performance(clf,features,poi)

        
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

Cs  = [1,10,100,1e3,1e4,1e5] 
gammas = [1e-5,1e-4,1e-3,1e-2,1e-1,1]
for c in Cs:
    for g in gammas:
        clf = SVC(kernel="rbf",gamma=g,C = c)
        print 'C:',c,'gamma:',g
        performance(clf,features,poi)
        

clf = SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

performance(clf,features,poi)


### without the new feature
features_list2 = ['poi', 'exercised_stock_options', 'expenses','salary']
data2 = featureFormat(my_dataset, features_list2, sort_keys = True)
my_poi2, my_features2 = targetFeatureSplit(data2)
features2 = np.array(my_features2)
poi2 = np.array(my_poi2)
performance(clf,features2,poi2)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
