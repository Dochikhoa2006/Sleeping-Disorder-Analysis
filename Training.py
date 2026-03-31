from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import re

random.seed (150)

dataset = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Sleeping Disorder/Sleep_health_and_lifestyle_dataset.csv')
data_dimension = dataset.shape
category_classifier = "Sleep Disorder"
one_hot_encoding_feature = []
ordered_target_encoding_feature = []


def missing_data ():
    dataset["Sleep Disorder"] = dataset["Sleep Disorder"].fillna ("Healthy")

def rate_into_categorial ():
    for index in range (dataset.shape[0]):

        rate = dataset.loc[index]["Blood Pressure"]
        pressure = re.split (r'/', rate)
        systolic = int (pressure[0])
        diastolic = int (pressure[1])
        
        if systolic < 120 and diastolic < 80:
            blood_pressure_category = 'normal'
        elif systolic <= 129 and systolic >= 120 and diastolic < 80:
            blood_pressure_category = 'elevated'
        elif (systolic <= 139 and systolic >= 130) or (diastolic <= 89 and diastolic >= 80):
            blood_pressure_category = 'stage 1 hypertension'
        elif systolic >= 140 or diastolic >= 90:
            blood_pressure_category = 'stage 2 hypertension'
        elif systolic >= 180 or diastolic >= 120: 
            blood_pressure_category = 'severe hypertension'
        else:
            blood_pressure_category = 'measurement error'
        
        dataset.at[index, "Blood Pressure"] = blood_pressure_category

def one_hot_encoding (feature, unique_category):
    for category in unique_category:
        dataset[category] = dataset[feature]
        dataset[category] = np.where (dataset[category] == category, 1, 0)

def ordered_target_encoding (feature, unique_category):
    output_feature_extract = dataset[category_classifier]
    output_feature_extract = list (output_feature_extract)
    
    temp = dataset[feature].copy ()
    random.shuffle (temp)
    alpha = 0.9

    for target in classifier:
        name_new_feature = feature + '_' + target
        dataset[name_new_feature] = 0.0
        
        total_appearance_i = output_feature_extract.count (target)
        overall_mean_i = total_appearance_i / data_dimension[0]  
        n_i = 0
        option_count = {category: 0 for category in unique_category} 

        for index in range (data_dimension[0]):
            current_category = temp[index]
            catboost_encode = (option_count[current_category] + overall_mean_i  * alpha) / (n_i + alpha)
            dataset.at[index, name_new_feature] = catboost_encode
            
            if output_feature_extract[index] == target:
                option_count[current_category] += 1
                n_i += 1

    return [option_count, overall_mean_i, alpha, n_i, classifier]


def categorial_into_numeric ():
    category_feature = dataset.select_dtypes (include = ['string'])
    del category_feature[category_classifier]
    
    threshold = 5
    for feature in category_feature:
        unique_category = dataset[feature].unique ()
        if len (unique_category) < threshold:
            one_hot_encoding (feature, unique_category)
            one_hot_encoding_feature.append ([feature, unique_category])
        else:
            ordered_target_encoding_scale = ordered_target_encoding (feature, unique_category)
            ordered_target_encoding_feature.append ([feature, ordered_target_encoding_scale])

        del dataset[feature]

def standardization ():
    for feature in dataset.columns:
        mean = dataset[feature].mean ()
        variance = np.var (dataset[feature])
        std = math.sqrt (variance)
        dataset[feature] = (dataset[feature] - mean) / std

def Logistic_and_SVM ():
    X_train, X_test, Y_train, Y_test = train_test_split (dataset_standardized, category_classifier_output, test_size = 0.22, random_state = 150)
    logistic = LogisticRegression (solver = 'lbfgs')
    logistic.fit (X_train, Y_train)
    predictions_LOGISTIC = logistic.predict (X_test)

    svm = SVC (kernel = 'poly', degree = 3, C = 1.0)
    svm.fit (X_train, Y_train)
    predictions_SVM = svm.predict (X_test)

    confusion = [confusion_matrix (predictions_SVM, Y_test), confusion_matrix (predictions_LOGISTIC, Y_test)]
    F1 = [precision_recall_f1 (confusion[0]), precision_recall_f1 (confusion[1])]
    confidence = [confidence_interval (predictions_LOGISTIC, Y_test), confidence_interval (predictions_SVM, Y_test)]
    
    return {"Softmax Regression": [F1[0], confidence[0]], "Support Vector Machine": [F1[1], confidence[1]]}

def Gradient_Boosting_Classifier ():
    X_train, X_test, Y_train, Y_test = train_test_split (dataset, category_classifier_output, test_size = 0.22, random_state = 150)
    gradient_boost = GradientBoostingClassifier (n_estimators = 80, max_depth = 2, random_state = 150)
    gradient_boost.fit (X_train, Y_train)
    predictions_GB = gradient_boost.predict (X_test)

    confusion = confusion_matrix (predictions_GB, Y_test)
    F1 = precision_recall_f1 (confusion)
    confidence = confidence_interval (predictions_GB, Y_test)

    return {"Gradient Boost": [F1, confidence]}, gradient_boost, X_test.columns.tolist ()

def confusion_matrix (predictions, Y_test):
    confusion_dict = {category: {temp: 0 for temp in classifier} for category in classifier}

    for index in range (len (Y_test)):
        predicted_category = predictions[index]
        observed_category = Y_test[index]
        confusion_dict[predicted_category][observed_category] += 1

    return confusion_dict

def precision_recall_f1 (confusion):
    f1 = {category: 0 for category in classifier}
    for category in classifier:
        TP = confusion[category][category]
        FP, FN = 0, 0
        for temp in classifier:
            if category != temp:
                FP += confusion[category][temp]
                FN += confusion[temp][category]

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1[category] = 2 * precision * recall / (precision + recall)
    
    return pd.Series (f1).mean ()

def confidence_interval (predictions, Y_test):
    acc = accuracy_score (Y_test, predictions)
    confidence = 0.95
    
    z_score = stats.norm.ppf ((1 + confidence) / 2)
    se = np.sqrt (acc * (1 - acc) / len (Y_test))
    margin_err = z_score * se 

    return [acc, margin_err]

def ploting (f1_and_confidence):
    model = ["Softmax Regression", "Support Vector Machine", "Gradient Boost"]
    f1 = [f1_and_confidence[model[0]][0], f1_and_confidence[model[1]][0], f1_and_confidence[model[2]][0]]
    confidence = [f1_and_confidence[model[0]][1], f1_and_confidence[model[1]][1], f1_and_confidence[model[2]][1]]
    for index in range (len (model)):
        f1[index] = round (float (f1[index]), 2)
        for i in range (int (2)):
            confidence[index][i] = round (float (confidence[index][i]), 2)
    accuracy_score = [confidence[0][0], confidence[1][0], confidence[2][0]]
    margin_error = [confidence[0][1], confidence[1][1], confidence[2][1]]

    fig, (graph1, graph2) = plt.subplots (1, 2, figsize = (15, 6))
    result_f1 = {'Model': model, 'F1 Score': f1}
    result_confidence = {'Model': model, 'Confidence Interval': confidence}

    sns.barplot (data = result_f1, x = 'Model', y = 'F1 Score', ax = graph1)
    graph1.set_title ("Comparision of Model F1 Score")
    graph1.set_xlabel ("Machine Learning Model")
    graph1.set_ylabel ("F1 Score")
    for p in graph1.patches:
        graph1.annotate (format (p.get_height (), '.2f'),
                            (p.get_x () + p.get_width () / 2., p.get_height ()),
                            ha = 'center', 
                            va = 'center',
                            xytext = (0, 9),
                            textcoords = 'offset points')
    
    graph2.errorbar (model, accuracy_score, yerr = margin_error, fmt = 'o', color = 'blue', ecolor = 'red')
    graph2.set_title ("Stability of Model Prediction")
    graph2.set_xlabel ("Machine Learning Model")
    graph2.set_ylabel ("Accuracy")

    plt.tight_layout (pad = 4.0)
    plt.savefig ("Graph.png")
    plt.show ()


missing_data ()
classifier = dataset[category_classifier].unique ()

rate_into_categorial ()
categorial_into_numeric ()
data_dimension = dataset.shape
category_classifier_output = np.array (dataset[category_classifier])
del dataset[category_classifier]

standardization ()
pca = PCA (n_components = 0.95)
dataset_standardized = pca.fit_transform (dataset)

f1_confidence_1 = Logistic_and_SVM ()
f1_confidence_2, model, model_features = Gradient_Boosting_Classifier ()
f1_and_confidence = {**f1_confidence_1, **f1_confidence_2}

ploting (f1_and_confidence)

joblib.dump (model, "Gradient Boosting Classifier.pkl")
joblib.dump (model_features, "Model Features.pkl")
joblib.dump (one_hot_encoding_feature, "One-Hot Encoding.pkl")
joblib.dump (ordered_target_encoding_feature, "Ordered Target Encoding.pkl")





# cd "/Users/chikhoado/Desktop/PROJECTS/Sleeping Disorder"
# python3 -m venv .venv
# source .venv/bin/activate
# pip install pandas scikit-learn numpy seaborn
# python "/Users/chikhoado/Desktop/PROJECTS/Sleeping Disorder/Training.py"