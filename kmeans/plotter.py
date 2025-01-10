import matplotlib.pyplot as plt

class ScatterPlotter:

  def __init__(self):
    self.cluster_centers = {}

  def plotClusteringData(self):
    x1, x2 = self.loadData('ClusteringData.txt')

    plt.figure(1)
    plt.scatter(x1, x2, color='blue')
    plt.title("Scatter Plot of (x1, x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

  def plotExamplesWithCenters(self):
    # Load the clustering data
    clustering_x1, clustering_x2 = self.loadData('ClusteringData.txt')

    # Load the discovered centers for the best run of each M group
    self.cluster_centers = {
      4: self.loadData('ClusterCenters_4.txt'),
      6: self.loadData('ClusterCenters_6.txt'),
      8: self.loadData('ClusterCenters_8.txt'),
      10: self.loadData('ClusterCenters_10.txt'),
      12: self.loadData('ClusterCenters_12.txt')
    }

    M_values = [4, 6, 8, 10, 12]

    # Create a scatter plot for each X value
    for X in M_values:
      # Select the appropriate cluster centers data based on X
      centers_x1, centers_x2 = self.cluster_centers[X]
      
      # Create a scatter plot for clustering data
      plt.scatter(clustering_x1, clustering_x2, marker='+', label=f'Examples')
      
      # Create a scatter plot for cluster centers
      plt.scatter(centers_x1, centers_x2, marker='*', label=f'Centers')
      
      # Add labels and a title
      plt.xlabel("x1")
      plt.ylabel("x2")
      plt.title(f"Scatter Plot for M={X}")
      
      # Show the legend outside the plot area
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
      plt.tight_layout()

      # Save the plot to disk (as a PNG file)
      plot_filename = f"scatter_plot_M{X}.png"
      plt.savefig(plot_filename)
      print(f"Plot saved as {plot_filename}")

      # Show the plot (optional, can be omitted if you just want to save the file)
      plt.show()

      # Close the plot to prevent overlapping when creating the next one
      plt.close()

  def loadData(self, file_path):
    x1, x2 = [], []
    with open(file_path, 'r') as file:
      for line in file:
          values = line.split()
          x1.append(float(values[0]))
          x2.append(float(values[1]))
    return x1, x2

if __name__ == "__main__":
    plotter = ScatterPlotter()
    # plotter.plotClusteringData()
    plotter.plotExamplesWithCenters()
