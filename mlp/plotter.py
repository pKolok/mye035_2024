import matplotlib.pyplot as plt

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
    plt.colorbar(label='Category')
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
    plt.colorbar(label='Category')
    plt.show()

if __name__ == "__main__":
    plotter = ScatterPlotter()
    plotter.plotClassificationTrainData()
    plotter.plotClassificationTestData()
