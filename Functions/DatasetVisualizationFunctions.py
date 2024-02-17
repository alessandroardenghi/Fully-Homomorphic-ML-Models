import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def plot_histograms(data, n_plots):

    num_cols = min(n_plots, len(data.columns))
    num_rows = math.ceil(num_cols / 4)  # Calculate the number of rows based on the number of columns
    figsize = (12, 3 * num_rows)  # Figure size

    fig, axs = plt.subplots(nrows=num_rows, ncols=4, figsize=figsize)

    # Flatten the axs array to simplify indexing
    axs = axs.flatten()

    # Iterate over columns and plot histograms
    for i, col in enumerate(data.columns):
        ax = axs[i]  # Select the appropriate axis for each column
        data[col].hist(ax=ax)

        # Set plot title and axis labels
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        if i == n_plots - 1:
            break

    # Remove empty subplots
    if num_cols % 4 != 0:
        for j in range(num_cols, num_rows * 4):
            fig.delaxes(axs[j])

    # Adjust subplot spacing and display the plots
    plt.tight_layout()
    plt.show()
    
def clip_values(df, columns, lower, upper):
    df_clipped = df.copy()
    for i in range(len(columns)):
        df_clipped[columns[i]] = df_clipped[columns[i]].clip(lower = lower[i], upper = upper[i])
    return df_clipped
    
def dataset_preprocessing(dataset, feature_names, target_column, cols_to_process, clipping):
    
    X = dataset[feature_names]
    y = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    scaler = RobustScaler()
    scaler.fit(X_train[cols_to_process])
    X_train[cols_to_process] = scaler.transform(X_train[cols_to_process])
    X_test[cols_to_process] = scaler.transform(X_test[cols_to_process])
    
    X_train = pd.DataFrame(X_train, columns = feature_names)
    X_test = pd.DataFrame(X_test, columns = feature_names)
    
    if clipping: 
        lower = X_train[cols_to_process].quantile(0.10).values
        upper = X_train[cols_to_process].quantile(0.90).values
        X_train = clip_values(X_train, cols_to_process, lower, upper)
        X_test = clip_values(X_test, cols_to_process, lower, upper)
    minima = X_train[cols_to_process].min()
    maxima = X_train[cols_to_process].max()

    X_train[cols_to_process] = 2 * ((X_train[cols_to_process] - minima) /
                                (maxima - minima)) - 1

    X_test[cols_to_process] = 2 * ((X_test[cols_to_process] - minima) /
                                (maxima - minima)) - 1
    X_test[cols_to_process]= X_test[cols_to_process].clip(lower=-1, upper=1)

    X_train = X_train.round(2)
    X_test = X_test.round(2)
    X_train = X_train.applymap('{:.2f}'.format)
    X_test = X_test.applymap('{:.2f}'.format)
    
    return X_train, X_test, y_train, y_test