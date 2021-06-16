#----- Split data into trainning set and testing set before find grid -----#
from data_preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size = 0.2, random_state = 7)

list_lon = [index[0] for index in x_train]
list_lat = [index[1] for index in x_train]
list_lon_test = [index[0] for index in x_test]
list_lat_test = [index[1] for index in x_test]
min_lon = min(list_lon)
max_lon = max(list_lon)

min_lat = min(list_lat)
max_lat = max(list_lat)

all_predict = {}
prediction = {}
classifiers =  [DecisionTreeClassifier(), DecisionTreeClassifier(criterion="entropy"),MLPClassifier(),GaussianNB(),KNeighborsClassifier(n_neighbors=3),KNeighborsClassifier(n_neighbors=5),KNeighborsClassifier(n_neighbors=7),SVC()]
names =  ["Decision Tree", "Decision Tree(criterion='entropy')", "Multilayer Perceptron", "Gaussian process", "K-Nearest Neighbors(n=3)", "K-Nearest Neighbors(n=5)", "K-Nearest Neighbors(n=7)", "Support Vector Machine"]

def fit_model(name,clf, X, y, X_test, y_test):
    clf.fit(X, y)
    predict_ = clf.predict(X_test)
    all_predict.setdefault(name,[]).append(len(predict_))
    prediction.setdefault(name,[]).append(accuracy_score(y_test, predict_, normalize=False))

def get_clf(gridSize):
    creatGrid(min(list_lon), max(list_lon), min(list_lat), max(list_lat), gridSize)
    print('-'*70)
    print(str(gridSize).center(50))
    result = findGrid2(x_train, y_train, x_test, y_test)

    dict_xTrain = result[0]
    dict_xTest = result[1]
    dict_yTrain = result[2]
    dict_yTest = result[3] 
    for x_key, y_key in zip(dict_xTest, dict_yTest):
        if(x_key in dict_xTrain.keys() and y_key in dict_yTrain.keys()):
            for name,clf in zip(names,classifiers):
                if(name == "Decision Tree" or name == "Decision Tree(criterion='entropy')"):
                    fit_model(name, clf, dict_xTrain[x_key], dict_yTrain[y_key], dict_xTest[x_key], dict_yTest[y_key])
                    
                if(name == "Multilayer Perceptron"):
                    fit_model(name, clf, dict_xTrain[x_key], dict_yTrain[y_key], dict_xTest[x_key], dict_yTest[y_key])

                if(name == "Gaussian process"):
                    fit_model(name, clf, dict_xTrain[x_key], dict_yTrain[y_key], dict_xTest[x_key], dict_yTest[y_key])

                if(name == "K-Nearest Neighbors(n=3)" or name == "K-Nearest Neighbors(n=5)" or name == "K-Nearest Neighbors(n=7)"):
                    if(clf.n_neighbors <= len(dict_xTrain[x_key])):
                        fit_model(name, clf, dict_xTrain[x_key], dict_yTrain[y_key], dict_xTest[x_key], dict_yTest[y_key])

                if(name == "Support Vector Machine"):
                    if(len(set(dict_yTrain[y_key])) > 1):
                        fit_model(name, clf, dict_xTrain[x_key], dict_yTrain[y_key], dict_xTest[x_key], dict_yTest[y_key])

    longest_name = max(names, key=len)
    for name in names:
        score = sum(prediction[name])/sum(all_predict[name])*100
        print('{name:<{name_width}}{between}{score:>{score_width}}'.format(name=name, name_width=len(longest_name), between=' '*7, score=score, score_width=len('Accuracy score')))
    print('-'*70)

get_clf(0.001)
get_clf(0.0001)