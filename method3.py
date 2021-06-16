#----- K-fold Cross Validation -----#
from data_preprocess import *
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

n_splits = 5
kfold = KFold(n_splits, random_state=7, shuffle=True)

x_train =[]
x_test =[]
y_train =[]
y_test =[]
list_lon = []
list_lat = []
for train_index, test_index in kfold.split(feature):
    x_train.append([])
    x_test.append([])
    y_train.append([])
    y_test.append([])
    list_lon.append([])
    list_lat.append([])

for i,(train_index, test_index) in enumerate(kfold.split(feature)):
    for j in train_index:
        x_train[i].append(feature[j])
        list_lon[i].append(feature[j][0])
        list_lat[i].append(feature[j][1])
        y_train[i].append(target[j])
    for j in test_index:
        x_test[i].append(feature[j])
        y_test[i].append(target[j])

#--------------------------------------------------------------------------------------------#
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
    for i in range(len(x_train)):
        creatGrid(min(list_lon[i]), max(list_lon[i]), min(list_lat[i]), max(list_lat[i]), gridSize)
        if(i == len(x_train)-1):
            print('-'*70)
            print(str(gridSize).center(50), 'n_splits = ', n_splits)
        result = findGrid2(x_train[i], y_train[i], x_test[i], y_test[i])
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
