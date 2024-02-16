import pickle
from TrainingFunctions import accuracy_measure
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def total_comparison(X_train, X_test, y_train, y_test, phi, S, dataset_name, directory, degrees, windows):
    my_trees = []
    sklearn_trees = []
    x = []    
    for x_count, degree in enumerate(degrees):
        my_trees.append([])
        for y_count, window in enumerate(windows):
            my_trees[x_count].append([])
            for depth in range(1, 6):
                tree_path = directory + f'/{dataset_name}_{degree}_{window}_{depth}.pkl'
                decision_tree_model_pkl = open(tree_path, 'rb')
                v = pickle.load(decision_tree_model_pkl)
                accuracy, _ = accuracy_measure(v, X_test, y_test, phi, S, False)
                my_trees[x_count][y_count].append(accuracy)
    
    for depth in range(1, 6):
        clf = DecisionTreeClassifier(max_depth = depth)
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        sklearn_trees.append(accuracy_score(y_test, y_pred))
        x.append(depth)

    fig, axes = plt.subplots(1, len(degrees), figsize=(15, 4))
    for degree, ax in enumerate(axes):
        for n_window, window in enumerate(windows):
            ax.plot(x, my_trees[degree][n_window], label = f'Tree with a = {window}', linestyle = 'dotted', marker = 'o')  
        ax.plot(x, sklearn_trees, label = 'Sklearn Tree')
        ax.set_xlabel('Tree Depth')
        ax.set_ylabel('Accuracy(%)')
        ax.set_title('Iris')
        ax.legend()
    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (6,4)
    plt.show()
