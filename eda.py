import seaborn as sns
import matplotlib.pyplot as plt

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
