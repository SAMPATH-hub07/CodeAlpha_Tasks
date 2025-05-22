# ZIP Code EDA Project
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load sample dataset
df = sns.load_dataset('titanic')

# Basic info
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Visualize survival rate by sex
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
