import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_graph():
    # Read the CSV file
    df = pd.read_csv('../NewNew_SS_Train_Results_Last_2/result_model_1.csv')

    # Display the first few rows of the dataframe
    print(df.head())

    plt.figure(figsize=(9, 6))
    
    # Plot the data (Loss)
    plt.plot(df['epoch'], df['train_loss'], marker='o', linestyle='-', color='b', label='Train_Loss')
    plt.plot(df['epoch'], df['test_loss'], marker='o', linestyle='-', color='y', label='Test_Loss')
    plt.title('Training Result')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # # Plot the data (Accuracy)
    # plt.plot(df['epoch'], df['test_accuracy'], marker='o', linestyle='-', color='y', label='Test_Accuracy')
    # plt.title('Test Accuracy Result')
    # plt.xlabel('Epochs')
    # plt.ylabel('% Accuracy')

    # Add a legend and place it outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Show the plot
    plt.show()
    
# def plot_correlation_matrix():
#     # Step 1: Load the data from the CSV file
#     data = pd.read_csv('./Val_1_type_models/20_Umap.csv')  # Replace 'your_file.csv' with the path to your CSV file
    
#     data = data.iloc[:, 1:-1]

#     # Step 2: Calculate the correlation matrix
#     correlation_matrix = data.corr()

#     # Step 3: Plot the correlation matrix
#     plt.figure(figsize=(10, 8))  # Set the size of the plot
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # Plot the heatmap with correlation values
#     plt.title('Correlation Matrix')  # Add title
#     plt.show()  # Show the plot


def plot_confusion_matrix(array, labels, title="Confusion Matrix", cmap="Blues"):
    """
    Plot a confusion matrix.

    Parameters:
        array (list or array-like): Array of shape (2, 2) containing the confusion matrix values.
        labels (list): List of length 4 containing the labels for the quadrants.
        title (str): Title for the confusion matrix plot. Default is "Confusion Matrix".
        cmap (str or Colormap): Colormap to be used for the heatmap. Default is "coolwarm".

    Returns:
        None
    """
    # Reshape the array into a 2x2 matrix
    confusion_matrix = np.array(array).reshape(2, 2)

    # Create a DataFrame for the confusion matrix
    confusion_df = pd.DataFrame(confusion_matrix, index=labels[:2], columns=labels[2:])

    # Plot the confusion matrix with color temperature
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.xlabel("Predict")
    plt.ylabel("Ground Truth")
    plt.show()

    

if __name__ == '__main__':
    # Example values
    array = [12, 4, 11, 37]
    labels = ["Death", "Survive", "Death", "Survive"]
    
    # plot_correlation_matrix()
    plot_graph()
    
    # Plot the confusion matrix
    # plot_confusion_matrix(array, labels, title="Confusion Matrix")