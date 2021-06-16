#----- Split data into trainning set and testing set after find grid -----#
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

list_lon = [index[0] for index in feature]
list_lat = [index[1] for index in feature]

all_predict = {}
prediction = {}
classifiers =  [DecisionTreeClassifier(), DecisionTreeClassifier(criterion="entropy"),MLPClassifier(),GaussianNB(),KNeighborsClassifier(n_neighbors=3),KNeighborsClassifier(n_neighbors=5),KNeighborsClassifier(n_neighbors=7),SVC()]
names =  [ "Decision Tree","Decision Tree(criterion='entropy')","Multilayer Perceptron","Gaussian process", "K-Nearest Neighbors(n=3)", "K-Nearest Neighbors(n=5)", "K-Nearest Neighbors(n=7)", "Support Vector Machine"]

def fit_model(name,clf, X, y, X_test, y_test):
    clf.fit(X, y)
    predict_ = clf.predict(X_test)
    all_predict.setdefault(name,[]).append(len(predict_))
    prediction.setdefault(name,[]).append(accuracy_score(y_test, predict_, normalize=False))

def get_clf(gridSize):
    creatGrid(min(list_lon), max(list_lon), min(list_lat), max(list_lat), gridSize)
    print('-'*70)
    print(str(gridSize).center(50))
    result = findGrid1(feature,target)
    
    dict_feature = result[0]
    dict_target = result[1]
    if(len(dict_feature) == len(dict_target)):
        for i in dict_feature.keys():
            if(len(dict_feature[i]) > 1):
                x_train, x_test, y_train, y_test = train_test_split(dict_feature[i], dict_target[i], test_size = 0.2, random_state = 0)
            
                for name,clf in zip(names,classifiers):
                    if(name == "Decision Tree" or name == "Decision Tree(criterion='entropy')"):
                       fit_model(name, clf, x_train, y_train, x_test, y_test)
                        
                    if(name == "Multilayer Perceptron"):
                        fit_model(name, clf, x_train, y_train, x_test, y_test)

                    if(name == "Gaussian process"):
                        fit_model(name, clf, x_train, y_train, x_test, y_test)

                    if(name == "K-Nearest Neighbors(n=3)" or name == "K-Nearest Neighbors(n=5)" or name == "K-Nearest Neighbors(n=7)"):
                        if(clf.n_neighbors <= len(x_train) ):
                            fit_model(name, clf, x_train, y_train, x_test, y_test)

                    if(name == "Support Vector Machine"):
                        if(len(set(y_train)) > 1):
                           fit_model(name, clf, x_train, y_train, x_test, y_test)

    longest_name = max(names, key=len)
    for name in names:
        score = sum(prediction[name])/sum(all_predict[name])*100
        print('{name:<{name_width}}{between}{score:>{score_width}}'.format(name=name, name_width=len(longest_name), between=' '*7, score=score, score_width=len('Accuracy score')))
    print('-'*70)

get_clf(0.001)
get_clf(0.0001)