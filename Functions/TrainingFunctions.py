import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from Functions.PolyApproxFunctions import poly_approx, step_function
import time
from DecisionTreeModel import build_tree, MyTree
import pickle

# Function to transform the input data in the appropriate format
def training_preprocess(X_train, y_train):

  X = np.array(X_train.values, dtype = float)

  n_labels = np.unique(y_train).size
  n_features = X.shape[1]

  y_tr_np = y_train.values
  y_tr = y_tr_np.reshape(-1, 1)

  encoder = OneHotEncoder()

  one_hot_ytr = encoder.fit_transform(y_tr).toarray()

  return X, one_hot_ytr, n_labels, n_features

# Function to initialize the set of thresholds and W
def initialization(size_dataset, n_thresholds):
  S = np.linspace(-1, 1, n_thresholds)
  S = np.around(S, decimals=2)
  depth = 1
  W = np.ones(size_dataset)

  return S, depth, W, n_thresholds

# FHE friendly implementation of Gini Impurity Measure
def Gini(right, left, n_thresholds, n_features, X):
  total_side = np.zeros((X.shape[1], n_thresholds))
  I_Gini = np.zeros((X.shape[1], n_thresholds))
  sides = [right, left]

  for theta in range(n_thresholds):
    for i in range(n_features):
      Gini_temp = 0

      for side in sides:
        total_side[i][theta] = np.sum(side[i][theta])
        Gini_temp += (1 - np.sum((side[i][theta]/total_side[i][theta])**2)) * total_side[i][theta]

      I_Gini[i][theta] = Gini_temp
  min_index = np.unravel_index(np.argmin(I_Gini), I_Gini.shape)
  return min_index

# FHE implementation of Decision Tree training algorithm
def Tree_Train(X, y, W, depth, node, max_depth, n_thresholds, n_labels, n_features, phi, S):
    right = np.zeros((X.shape[1], n_thresholds, n_labels))
    left = np.zeros((X.shape[1], n_thresholds, n_labels))
    w_right = np.zeros(len(X))
    w_left = np.zeros(len(X))

    if depth == max_depth:
      temp = np.zeros(n_labels)
      index = np.argmax(np.dot(W, y))
      temp[index] = 1
      node.leaf_value = temp

    else:
      for i in range(n_features):
        for theta in range(n_thresholds):
          temp_r = np.zeros(n_labels)
          temp_l = np.zeros(n_labels)

          for x in range(len(X)):
            temp_r += W[x] * phi(X[x][i] - S[theta]) * y[x]
            temp_l += W[x] * phi(S[theta] - X[x][i]) * y[x]
          right[i][theta] = temp_r
          left[i][theta] = temp_l
      node.feature, node.threshold = Gini(right, left, n_thresholds, n_features, X)

      for x in range(len(X)):
        w_right[x] = W[x] * phi(X[x][node.feature] - S[node.threshold])
      Tree_Train(X, y, w_right, depth + 1, node.right, max_depth, n_thresholds, n_labels, n_features, phi, S)

      for x in range(len(X)):
        w_left[x] = W[x] * phi(S[node.threshold] - X[x][node.feature])
      Tree_Train(X, y, w_left, depth + 1, node.left, max_depth, n_thresholds, n_labels, n_features, phi, S)
  
# FHE Friendly implementation of Decision Tree prediction algorithm    
def Tree_Predict(node, x, phi, S):
  if node.leaf_value is not None:
    return node.leaf_value
  else:
    score = phi(x[node.feature] - S[node.threshold]) * Tree_Predict(node.right, x, phi, S) + phi(S[node.threshold] - x[node.feature]) * Tree_Predict(node.left, x, phi, S)
    return  score

# Function to provide measures on quality of prediction
def accuracy_measure(root, X_test, y_test, phi, S, switch):

  my_X_test = np.array(X_test.values, dtype = float)

  my_y_pred = []
  for x in my_X_test:
    my_y_pred.append(np.argmax(Tree_Predict(root, x, phi, S)))


  my_y_pred = np.array(my_y_pred)
  my_y_pred = np.array(["{:.2f}".format(float(x)) for x in my_y_pred])

  y_test = np.array(y_test.values, dtype = float)
  y_test = np.array(["{:.2f}".format(float(x)) for x in y_test])


  accuracy = accuracy_score(y_test, my_y_pred)
  cm = confusion_matrix(y_test, my_y_pred)


  if switch:
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)

  return accuracy, cm


def training_function(X, y, W, n_thresholds, S, degrees, windows, directory, dataset_name):
    n_labels = y.shape[1]
    n_features = X.shape[1]
    times = []
    for x_count, degree in enumerate(degrees):
        times.append([])
        
        for y_count, window in enumerate(windows):
            times[x_count].append([])
            phi = poly_approx(step_function, degree, 2, window)
            
            for i in range(1, 6):
                start_time = time.time()
                v = build_tree(i)
                Tree_Train(X, y, W, 1, v, i, n_thresholds, n_labels, n_features, phi, S)
                decision_tree_pkl_filename = directory + f'{dataset_name}_{degree}_{window}_{i}.pkl'
                decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
                pickle.dump(v, decision_tree_model_pkl)

                decision_tree_model_pkl.close()
                end_time = time.time()
                t = round(end_time - start_time, 3)
                times[x_count][y_count].append(t)
        
        print(f'Training of Trees of degree {degree} completed')
    return times