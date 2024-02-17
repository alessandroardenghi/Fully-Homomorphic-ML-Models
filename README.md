# Privacy Preseving Decision Trees

<!-- ABOUT THE PROJECT -->
## About The Project

This project is based on the paper "Privacy-Preserving Decision Trees Training and Predictions", published by Akavia et al. in 2022. In this paper, the authors describe a new protocol to train tree-based models and perform prediction on data encrypted using the CKKS encryption scheme. <br>
The privacy-preserving protocol implemented in this project works as follows: there is a Client who owns some private data and a Server who owns a tree-based model and some data of its own. The Server will first train its model on plaintext data, and then the Client will encrypt their private data using CKKS and send the encrypted data to the Server. The Server will then run the ML model on it and return the encrypted result to the Client, who is then free to decrypt it. <br>
The key concept of this protocol is that, although the Server's model works on the Client's data, it will be impossible for the Server to gain any information from this interaction, specifically because the data is encrypted with CKKS, and so will be the output of the model. Nonetheless, the result of the computations will be the same as if the computations were carried out on plaintext data. <br>
The project is structured in the following way:
- **Polynomial** Approximation: since evaluating functions like tanh, ReLU and sign on encrypted data, the authors suggest substituting the hard threshold function present in the training protocols of traditional decision trees with a polynomial approximation.
- **Data Exploration and Cleaning**
- **Model Selection**: in this section, I implement the training protocol suggested by the authors and choose the best models
- **Prediction on Encrypted data**: in this section, I perform predictions on the encrypted data using the prediction protocol suggested by the authors, and I compare the results obtained by the prediction on encrypted data with the results obtained by the predictions on plaintext data.
- **Conclusions**
