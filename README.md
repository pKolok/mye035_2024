# Machine Learning Implementations in C

This project contains C implementations of two fundamental machine learning algorithms: K-Means Clustering and a Multi-Layer Perceptron (MLP) for classification. Each implementation includes a data generator and a Python script for visualizing the results.

## Prerequisites

Before you begin, ensure you have the following installed:

- **GCC (GNU Compiler Collection):** To compile the C code.
- **Python 3:** To run the visualization scripts.
- **Python Libraries:**
  - `matplotlib`
  - `numpy`

You can install the required Python libraries using pip:
```bash
pip install matplotlib numpy
```

---

## 1. K-Means Clustering

This part of the project implements the K-Means clustering algorithm from scratch in C. The program generates a synthetic 2D dataset and groups it into a specified number of clusters.

### How it Works

- **Data Generation:** A C function generates 1000 2D data points with some inherent clustering. This data is saved to `ClusteringData.txt`.
- **K-Means Algorithm:** The C implementation of K-Means runs 20 times and selects the best clustering (the one with the minimum error). The number of clusters `M` is hardcoded to 12 in `kmeans/KMeansBuilder.c`.
- **Output:** The final cluster centers are saved to a file named `ClusterCenters_M.txt` (e.g., `ClusterCenters_12.txt`).

### How to Run

1.  **Compile the C code:**
    Open a terminal and navigate to the project's root directory. Use the following command to compile the K-Means program:
    ```bash
    gcc kmeans/main.c kmeans/KMeansBuilder.c kmeans/dataGenerator.c -o kmeans/kmeans -lm
    ```
    The `-lm` flag is necessary to link the math library.

2.  **Run the executable:**
    ```bash
    ./kmeans/kmeans
    ```
    This will generate the `ClusteringData.txt` and `ClusterCenters_12.txt` files in the root directory.

### Visualization

A Python script is provided to visualize the results.

1.  **Run the plotter:**
    ```bash
    python3 kmeans/plotter.py
    ```
    This script will:
    - Read the `ClusteringData.txt` and `ClusterCenters_12.txt` files.
    - Generate a scatter plot showing the data points and the final cluster centers.
    - Save the plot as an image file (e.g., `scatter_plot_M12.png`).
    - Display the plot on the screen.

    *Note: The plotter is configured to look for center files for M = 4, 6, 8, 10, and 12. To generate all plots, you will need to modify the `M` macro in `kmeans/KMeansBuilder.c`, recompile, and run the C program for each value of M.*

---

## 2. Multi-Layer Perceptron (MLP)

This part of the project implements and compares two Multi-Layer Perceptrons for a 4-class classification problem. The networks are built from scratch in C.

### How it Works

- **Data Generation:** A C function generates a non-linearly separable 2D dataset with 4000 training points and 4000 test points. The data is saved to `ClassificationTrainData.txt` and `ClassificationTestData.txt`.
- **Network Architectures:**
  - **2-Hidden-Layer MLP:**
    - Input Layer: 2 neurons
    - Hidden Layer 1: 15 neurons (`tanh` activation)
    - Hidden Layer 2: 15 neurons (`relu` activation)
    - Output Layer: 4 neurons (`softmax` activation)
  - **3-Hidden-Layer MLP:**
    - Input Layer: 2 neurons
    - Hidden Layer 1: 10 neurons (`tanh` activation)
    - Hidden Layer 2: 10 neurons (`tanh` activation)
    - Hidden Layer 3: 10 neurons (`relu` activation)
    - Output Layer: 4 neurons (`softmax` activation)
- **Training:** The networks are trained using backpropagation with a batch size of 20 and a learning rate of 0.001.
- **Output:** The program prints the classification accuracy on the test set and saves the detailed results (including which points were classified correctly) to `ClassificationTestResults.txt`.

### How to Run

1.  **Compile the C code:**
    From the project's root directory, run:
    ```bash
    gcc mlp/main.c mlp/ModelBuilderPT2.c mlp/ModelBuilderPT3.c mlp/dataGenerator.c -o mlp/mlp -lm
    ```

2.  **Run the executable:**
    ```bash
    ./mlp/mlp
    ```
    This will train and test both networks and generate the data and results files in the root directory.

### Visualization

A Python script is provided to visualize the dataset and the classification results.

1.  **Run the plotter:**
    ```bash
    python3 mlp/plotter.py
    ```
    This script will generate and display three plots:
    1.  A scatter plot of the training data, color-coded by class.
    2.  A scatter plot of the test data, color-coded by class.
    3.  A scatter plot of the test results, showing the true class by color and whether the prediction was correct (`+`) or incorrect (`o`). This plot provides a clear visual representation of the model's performance.
