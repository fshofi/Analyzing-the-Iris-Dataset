# Analyzing the Iris Dataset

This project explores the famous Iris dataset using data analysis and machine learning techniques. The Iris dataset is a well-known dataset in the field of machine learning and contains 150 observations of iris flowers with four features: sepal length, sepal width, petal length, and petal width. Each observation is classified into one of three species: setosa, versicolor, or virginica.

## Project Structure

The project is organized into the following files:

- `data_preparation.py`: Script for loading and preparing the dataset.
- `eda.py`: Script for performing exploratory data analysis (EDA) and visualizing the data.
- `model_training.py`: Script for training and evaluating a RandomForest classifier.
- `DataScienceIris.ipynb`: Jupyter notebook that combines all steps of the project, including data loading, EDA, and model training.

## Project Steps

### 1. Data Preparation
The `data_preparation.py` script includes:
- Loading the Iris dataset from the seaborn library.
- Displaying summary statistics.
- Checking for missing values.
- Displaying the first few rows of the dataset.

### 2. Exploratory Data Analysis (EDA)
The `eda.py` script includes:
- Generating summary statistics for numeric and categorical columns.
- Plotting histograms for numeric columns to visualize their distributions.
- Creating a pairplot to explore relationships between features.
- Visualizing a correlation matrix to identify correlations among numeric features.

### 3. Predictive Analysis
The `model_training.py` script includes:
- Splitting the data into training and testing sets.
- Training a RandomForest classifier on the training set.
- Predicting species on the test set.
- Evaluating the model's performance using accuracy and a classification report.

## Results
The project provided insights into the Iris dataset's structure and relationships among features. The RandomForest classifier achieved high accuracy in predicting iris species, demonstrating the effectiveness of machine learning techniques on this dataset.

## Conclusion
This project showcases the application of data analysis and machine learning techniques to the Iris dataset. The analysis and model demonstrated strong performance, providing valuable insights and accurate predictions.

## Repository
The complete project, including all Python scripts and the Jupyter notebook, is available in this repository.

## How to Run the Scripts
To run the scripts in this project, follow these steps:
1. Clone the repository to your local machine.
2. Ensure you have the necessary Python libraries installed (pandas, seaborn, matplotlib, scikit-learn).
3. Run the Python scripts in the following order:
   - `data_preparation.py`
   - `eda.py`
   - `model_training.py`

## GitHub Repository URL
[Your GitHub URL]

Feel free to explore the repository and use the provided scripts for your own analysis and machine learning projects.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
