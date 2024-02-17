from DecisionTreeModel import MyTree, build_tree, print_tree
import numpy as np
from Functions.TrainingFunctions import training_preprocess, initialization, Tree_Predict
import pickle
import time
from Functions.EncryptionFunctions import encrypt_1d, Enc_Predict
from sklearn.metrics import confusion_matrix, accuracy_score

class RandomForest:
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.estimators = []
        self.max_depth = max_depth
        for i in range(n_estimators):
          v = build_tree(max_depth)
          self.estimators.append(v)
          

def print_rf(forest):
  for i, tree in enumerate(forest.estimators):
    print(f"\nEstimator {i}")
    print_tree(tree)
    
def update_feature(tree, feature_list):
  if tree.leaf_value is None:
    tree.feature = feature_list[tree.feature]
    update_feature(tree.right, feature_list)
    update_feature(tree.left, feature_list)
    
def rf_train_subroutine(X, y, W, depth, node, max_depth, n_labels, n_features, S, phi):
    right = np.zeros((X.shape[1], 50, n_labels))
    left = np.zeros((X.shape[1], 50, n_labels))
    w_right = np.zeros(len(X))
    w_left = np.zeros(len(X))

    if depth == max_depth:
      temp = np.zeros(n_labels)
      index = np.argmax(np.dot(W, y))
      temp[index] = 1
      node.leaf_value = temp

    else:
      for i in range(n_features):
        for theta in range(50):
          temp_r = np.zeros(n_labels)
          temp_l = np.zeros(n_labels)

          for x in range(len(X)):
            temp_r += W[x] * phi(X[x][i] - S[theta]) * y[x]
            temp_l += W[x] * phi(S[theta] - X[x][i]) * y[x]
          right[i][theta] = temp_r
          left[i][theta] = temp_l

      node.feature, node.threshold = rf_Gini(right, left, X, n_features)
     
      for x in range(len(X)):
        w_right[x] = W[x] * phi(X[x][node.feature] - S[node.threshold])
      rf_train_subroutine(X, y, w_right, depth + 1, node.right, max_depth, n_labels, n_features, S, phi)

      for x in range(len(X)):
        w_left[x] = W[x] * phi(S[node.threshold] - X[x][node.feature])
      rf_train_subroutine(X, y, w_left, depth + 1, node.left, max_depth, n_labels, n_features, S, phi)
      
def rf_Gini(right, left, X, n_features):
  total_side = np.zeros((X.shape[1], 50))
  I_Gini = np.zeros((X.shape[1], 50))
  sides = [right, left]

  for theta in range(50):
    for i in range(n_features):
      Gini_temp = 0

      for side in sides:
        total_side[i][theta] = np.sum(side[i][theta])
        Gini_temp += (1 - np.sum((side[i][theta]/total_side[i][theta])**2)) * total_side[i][theta]

      I_Gini[i][theta] = Gini_temp
  min_index = np.unravel_index(np.argmin(I_Gini), I_Gini.shape)
  return min_index

def RF_Train(rf, X_train, y_train, phi, filename):
    X, y, n_labels, n_features = training_preprocess(X_train, y_train)
    n_sampled_features = int(np.sqrt(n_features))
    S, depth, W, n_thresholds = initialization(len(X), 50)
    #n_sampled_features = int(2/3 * n_features)
    for tree in rf.estimators:
        indices = np.random.choice(len(X), size=len(X), replace=True)
        feature_indices = np.random.choice(n_features, size = n_sampled_features, replace=False)
        X_new = X[indices][:, feature_indices]
        y_new = y[indices]

        rf_train_subroutine(X_new, y_new, W, depth, tree, rf.max_depth, n_labels, n_sampled_features, S, phi)

        update_feature(tree, feature_indices)
    model = open(filename, 'wb')
    pickle.dump(rf, model)
    model.close()

def RF_Predict(rf, x, n_labels, phi, S):
  prediction = np.zeros((rf.n_estimators, n_labels))
  for i, tree in enumerate(rf.estimators):
    prediction[i] = Tree_Predict(rf.estimators[i], x, phi, S)
  final_score = np.sum(prediction, axis = 0)
  return final_score/rf.n_estimators

def RF_Enc_Predict(HE, rf, x_enc, n_labels, phi, S):
    
  score = Enc_Predict(HE, rf.estimators[0], x_enc, n_labels, phi, S)
  for tree in rf.estimators[1:]:
    prediction = Enc_Predict(HE, tree, x_enc, n_labels, phi, S)
    score += prediction

  return score

def RF_Multiple_Enc_Predict(HE, X, X_test, phi, rf, n_labels):
  initial_time = time.time()
  S, _, _, _ = initialization(len(X), 50)
  my_X_test = np.array(X_test.values, dtype = float)

  encrypted_predictions = []
  secure_predictions = []
  counter = 0
  for x in my_X_test:
    counter += 1
    x_enc = encrypt_1d(HE, x, n_labels)
    prediction = RF_Enc_Predict(HE, rf, x_enc, n_labels, phi, S)
    encrypted_predictions.append(prediction)
    secure_predictions.append(np.argmax(HE.decryptFrac(prediction)))
    if counter % 20 == 0:
      print(f"{counter}/{len(my_X_test)} Predictions Completed")
  final_time = time.time()
  total_time = final_time - initial_time
  print(f"Prediction Time = {total_time: .2f} s")
  
  return total_time, encrypted_predictions, secure_predictions


def rf_accuracy_measure(rf, X_train, y_train, X_test, y_test, phi):

    _, _, n_labels, _ = training_preprocess(X_train, y_train)
    S, _, _, _ = initialization(len(X_train), 50)  
    my_X_test = np.array(X_test.values, dtype = float)

    my_y_pred = []
    for x in my_X_test:
        my_y_pred.append(np.argmax(RF_Predict(rf, x, n_labels, phi, S)))


    my_y_pred = np.array(my_y_pred)
    my_y_pred = np.array(["{:.2f}".format(float(x)) for x in my_y_pred])

    y_test = np.array(y_test.values, dtype = float)
    y_test = np.array(["{:.2f}".format(float(x)) for x in y_test])

    accuracy = accuracy_score(y_test, my_y_pred)
    cm = confusion_matrix(y_test, my_y_pred)

    print(f"Accuracy: {accuracy: .2f}")
    print("Confusion Matrix:\n", cm)

