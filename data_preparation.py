# Write data_preparation.py
data_preparation_code = """
import pandas as pd
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display summary statistics
print(iris.describe())

# Check for missing values
print(iris.isnull().sum())

# Display the first few rows
print(iris.head())
"""

with open("data_preparation.py", "w") as file:
    file.write(data_preparation_code)

# Write eda.py
eda_code = """
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Function to suppress specific warnings during plotting
def suppress_specific_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Glyph.*missing from font")
            warnings.filterwarnings("ignore", message="Matplotlib currently does not support Arabic natively")
            warnings.filterwarnings("ignore", message="Font 'default' does not have a glyph")
            return func(*args, **kwargs)
    return wrapper

# Plot histograms for numeric columns with adjusted layout
@suppress_specific_warnings
def plot_histograms():
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
    axes = axes.flatten()
    for ax, col in zip(axes, iris.select_dtypes(include=['number']).columns):
        iris[col].hist(bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

plot_histograms()

# Plot pairplot
sns.pairplot(iris, hue='species')
plt.show()

# Plot correlation matrix
@suppress_specific_warnings
def plot_correlation_matrix():
    plt.figure(figsize=(12, 8))
    sns.heatmap(iris.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

plot_correlation_matrix()
"""

with open("eda.py", "w") as file:
    file.write(eda_code)

# Write model_training.py
model_training_code = """
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Prepare the data
X = iris.drop(columns=['species'])
y = iris['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
"""

with open("model_training.py", "w") as file:
    file.write(model_training_code)

# Verify files were created
!ls -l *.py
