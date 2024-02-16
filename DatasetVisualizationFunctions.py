import math
import matplotlib.pyplot as plt
def plot_histograms(data):

    num_cols = len(data.columns)
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

    # Remove empty subplots
    if num_cols % 4 != 0:
        for j in range(num_cols, num_rows * 4):
            fig.delaxes(axs[j])

    # Adjust subplot spacing and display the plots
    plt.tight_layout()
    plt.show()