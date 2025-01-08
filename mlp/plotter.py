import matplotlib.pyplot as plt
import numpy as np

class ScatterPlotter:

  def plotClassificationTrainData(self):
    x1 = []; x2 = []; categories = []

    with open('ClassificationTrainData.txt', 'r') as file:
      for line in file:
        values = line.split()
        x1.append(float(values[0]))
        x2.append(float(values[1]))
        categories.append(int(values[2]))

    # Create a scatter plot
    plt.figure(1)
    plt.scatter(x1, x2, c=categories, cmap='viridis', edgecolor='k')
    plt.title("Scatter Plot of (x1, x2) Train Data Set")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

  def plotClassificationTestData(self):
    x1 = []; x2 = []; categories = []

    with open('ClassificationTestData.txt', 'r') as file:
      for line in file:
        values = line.split()
        x1.append(float(values[0]))
        x2.append(float(values[1]))
        categories.append(int(values[2]))

    # Create a scatter plot
    plt.figure(2)
    plt.scatter(x1, x2, c=categories, cmap='viridis', edgecolor='k')
    plt.title("Scatter Plot of (x1, x2) Test Data Set")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

  def plotClassificationTestResults(self):
    x1 = []; x2 = []; categories = []; predicted = []

    with open('ClassificationTestResults.txt', 'r') as file:
      for line in file:
        values = line.split()
        x1.append(float(values[0]))
        x2.append(float(values[1]))
        categories.append(int(values[2]))
        predicted.append(int(values[3]))

    # Convert lists to NumPy arrays for easier masking
    x1 = np.array(x1)
    x2 = np.array(x2)
    categories = np.array(categories)
    predicted = np.array(predicted)

    # Define a color map for categories
    colors = {1: 'purple', 2: 'green', 3: 'blue', 4: 'red'}

    # Create the scatter plot
    plt.figure(figsize=(9, 6))

    for category in [1,2,3,4]:
      for prediction in [0, 1]:
        # Filter points by category and prediction
        mask = (categories == category) & (predicted == prediction)
        marker = '+' if prediction == 1 else 'o'
        plt.scatter(x1[mask], x2[mask], color=colors[category],
                    marker=marker, label=f'C = {category} - Pred = {prediction}')

    # Add legend and labels
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Legend')
    plt.tight_layout(pad=3.0)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter Plot with Categories and Predictions')
    plt.show()

if __name__ == "__main__":
    plotter = ScatterPlotter()
    plotter.plotClassificationTrainData()
    plotter.plotClassificationTestData()
    plotter.plotClassificationTestResults()
