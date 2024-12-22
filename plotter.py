import matplotlib.pyplot as plt

class ScatterPlotter:

  def plotClassificationData(self):
    x1 = []; x2 = []; categories = []

    with open('ClassificationData.txt', 'r') as file:
      for line in file:
        values = line.split()
        x1.append(float(values[0]))
        x2.append(float(values[1]))
        categories.append(int(values[2]))

    # Create a scatter plot
    plt.figure(1)
    plt.scatter(x1, x2, c=categories, cmap='viridis', edgecolor='k')
    plt.title("Scatter Plot of (x1, x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label='Category')
    plt.show()

  def plotClusteringData(self):
    x1 = []; x2 = []

    with open('ClusteringData.txt', 'r') as file:
      for line in file:
        values = line.split()
        x1.append(float(values[0]))
        x2.append(float(values[1]))

    plt.figure(2)
    plt.scatter(x1, x2, color='blue')
    plt.title("Scatter Plot of (x1, x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

if __name__ == "__main__":
    plotter = ScatterPlotter()
    plotter.plotClassificationData()
    plotter.plotClusteringData()
