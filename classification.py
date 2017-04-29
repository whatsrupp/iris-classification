import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def initial_data_load():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    return [dataset, names]

main_dataset = initial_data_load()[0]
names = initial_data_load()[1]

def load_models():
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    return models

def run_all_models():
    models = load_models()


    datasets = create_validation()
    X_train = datasets['X_train']
    Y_train = datasets['Y_train']
    X_validation = datasets['X_validation']
    Y_validation = datasets['X_validation']
    results = []
    names = []

    seed = 7
    scoring = 'accuracy'

    for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "Accuracy of %s cross-validation method: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    plot_cross_validation_results(results)

def plot_cross_validation_results(results):
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def data_summary(dataset):
    print(dataset.describe())

def plot_box_and_whisker(dataset):
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

def plot_histogram(dataset):
    dataset.hist()
    plt.show()


def plot_multivariate_scatter(dataset):
    scatter_matrix(dataset)
    plt.show()

def create_validation():
    array = main_dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    datasets = {}
    datasets['X_train']= X_train
    datasets['Y_train'] = Y_train
    datasets['X_validation'] = X_validation
    datasets['X_validation'] = Y_validation
    return datasets
