import numpy as np
import time
from Functions.TrainingFunctions import initialization
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

def encode_2d(HE, array, n_labels):
  array_encoded = []
  for x in range(len(array)):
    array_encoded.append([])
    for feature in range(len(array[x])):
      value_to_encode = np.array([array[x][feature] for i in range(n_labels)], dtype = np.float64)
      plaintext_value = HE.encodeFrac(value_to_encode)
      array_encoded[x].append(plaintext_value)
  return array_encoded


def encrypt_2d(HE, array, n_labels):
  array_encrypted = []
  encoded_array = encode_2d(array, n_labels)
  for x in range(len(encoded_array)):
    array_encrypted.append([])
    for feature in range(len(encoded_array[x])):
      ciphertext_value = HE.encryptPtxt(encoded_array[x][feature])
      array_encrypted[x].append(ciphertext_value)
  return array_encrypted


def encode_1d(HE, array, n_labels):
  encoded_array = []
  for i in range(len(array)):
    coefficient = np.array([array[i] for j in range(n_labels)], dtype = np.float64)
    #print(coefficient)
    encoded_coefficient = HE.encodeFrac(coefficient)
    encoded_array.append(encoded_coefficient)
  return encoded_array


def encrypt_1d(HE, array, n_labels):
  encrypted_array = []
  encoded_array = encode_1d(HE, array, n_labels)
  for i in range(len(encoded_array)):
    encrypted_array.append(HE.encryptPtxt(encoded_array[i]))
  return encrypted_array


def apply_poly(HE, polynomial, ciphertext, n_labels):

  l = [ciphertext]
  encrypted_poly = encrypt_1d(HE, np.flip(polynomial.c), n_labels)
  for i in range(len(encrypted_poly) - 2):
    c = ~(l[i] * ciphertext)
    l.append(c)
  result = encrypted_poly[0]
  result += np.sum(np.array(l) * np.array(encrypted_poly[1:]))

  return result


def Enc_Predict(HE, v, cipher, n_labels, phi, S):
  if v.leaf_value is None:
    encrypted_threshold = encrypt_1d(HE, [S[v.threshold]], n_labels)[0]
    arg1 = cipher[v.feature] - encrypted_threshold
    arg2 = encrypted_threshold - cipher[v.feature]
    coeff1 = ~(apply_poly(HE, phi, arg1, n_labels))
    coeff2 = ~(apply_poly(HE, phi, arg2, n_labels))
    vector1 = Enc_Predict(HE, v.right, cipher, n_labels, phi, S)
    vector2 = Enc_Predict(HE, v.left, cipher, n_labels, phi, S)
    label_vector = ~(coeff1 * vector1) + ~(coeff2 * vector2)
    return label_vector
  else:
    encrypted_leaf_value = HE.encrypt(v.leaf_value)
    return encrypted_leaf_value
  
def Multiple_Enc_Predict(HE, X, X_test, phi, tree, n_labels):
  initial_time = time.time()
  S, _, _, _ = initialization(len(X), 50)
  my_X_test = np.array(X_test.values, dtype = float)

  encrypted_predictions = []
  secure_predictions = []
  counter = 0
  for x in my_X_test:
    counter += 1
    x_enc = encrypt_1d(HE, x, n_labels)
    prediction = Enc_Predict(HE, tree, x_enc, n_labels, phi, S)
    encrypted_predictions.append(prediction)
    secure_predictions.append(np.argmax(HE.decryptFrac(prediction)))
    if counter % 10 == 0:
      print(f"{counter}/{len(my_X_test)} Predictions Completed")
  final_time = time.time()
  total_time = final_time - initial_time
  print(f"Prediction Time = {total_time: .2f}")
  
  return total_time, encrypted_predictions, secure_predictions

def save_predictions(filename, file):
  
  model = open(filename, 'wb')
  pickle.dump(file, model)
  model.close()
  
def prediction_accuracy(decrypted_predictions, y_test):
  
  my_y_pred = np.array(decrypted_predictions)
  my_y_pred = np.array(["{:.2f}".format(float(x)) for x in my_y_pred])

  y_test = np.array(y_test.values, dtype = float)
  y_test = np.array(["{:.2f}".format(float(x)) for x in y_test])


  accuracy = accuracy_score(y_test, my_y_pred)
  cm = confusion_matrix(y_test, my_y_pred)
  print(f"Accuracy: {accuracy: .2f}")
  print(f"Confusion Matrix:\n {cm}")