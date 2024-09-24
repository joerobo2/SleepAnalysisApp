# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('cmu-sleep.csv')

# Confirm dataset is loaded
df.head()

# Confirm dataset details

print(f'\nDataset Info: \n {df.info()}')
print(f'\nSummary Statistics: \n {df.describe()}')
print(f'\nUnqiue Values: \n {df.nunique()}')
print(f'\nRows and Cols: \n {df.shape}')
print(f'\n Column Names: \n {df.columns}')
print(f'\n Data Types: \n {df.dtypes}')
print(f'\n Missing Values: \n {df.isnull().sum()}')

# Confirm categorical variables
df.select_dtypes(include=['object']).info()
print(f'\nCohort: \n{df["cohort"].value_counts()}')
print(f'\nGender: \n{df["demo_gender"].value_counts()}')
print(f'\nRace: \n{df["demo_race"].value_counts()}')
print(f'\nFirstGen: \n{df["demo_firstgen"].value_counts()}')

# Change Term units and Z Term Units of Z to numerical
# Replace empty strings or spaces with NaN

df['term_units'] = df['term_units'].replace(' ', np.nan)
df['Zterm_units_ZofZ'] = df['Zterm_units_ZofZ'].replace(' ', np.nan)

# Convert the column to float

df['term_units'] = df['term_units'].astype(float)
df['Zterm_units_ZofZ'] = df['Zterm_units_ZofZ'].astype(float)

# Confirm numerical variables

df.select_dtypes(include=['int64', 'float64']).info()
df.select_dtypes(include=['int64', 'float64']).describe().T

# Univariate Analysis
# Categorical Variables: Analyze the distribution of categorical variables (demo_gender, demo_race, demo_firstgen, cohort, term_units, ) using freq tables and charts
# There are missing and inaccurate values for each variable that need to be removed before conducting further analysis

# Gender
print(f'\nGender: \n{df["demo_gender"].value_counts()}')
print(f'\nGender: \n{df["demo_gender"].value_counts(normalize=True)*100}')
sns.countplot(x='demo_gender', data=df)
plt.title('Gender Distribution')
plt.show()

# FirstGen
print(f'\nFirstGen: \n{df["demo_firstgen"].value_counts()}')
print(f'\nFirstGen: \n{df["demo_firstgen"].value_counts(normalize=True)*100}')
sns.countplot(x='demo_firstgen', data=df)
plt.title('FirstGen Distribution')
plt.show()

# Race
print(f'\nRace: \n{df["demo_race"].value_counts()}')
print(f'\nRace: \n{df["demo_race"].value_counts(normalize=True)*100}')
sns.countplot(x='demo_race', data=df)
plt.title('Race Distribution')
plt.show()

# Cohort
print(f'\nCohort: \n{df["cohort"].value_counts()}')
print(f'\nCohort: \n{df["cohort"].value_counts(normalize=True)*100}')
sns.countplot(x='cohort', data=df)
plt.title('Cohort Distribution')
plt.show()

# Remove missing and inaccurate information from categorical variables
df = df.dropna(subset=['demo_gender', 'demo_firstgen', 'demo_race'])
df = df[df['demo_gender'].isin(['0', '1'])]
df = df[df['demo_firstgen'].isin(['0', '1'])]
df = df[df['demo_race'].isin(['0', '1'])]

# Gender
print(f'\nGender: \n{df["demo_gender"].value_counts()}')
print(f'\nGender: \n{df["demo_gender"].value_counts(normalize=True)*100}')
sns.countplot(x='demo_gender', data=df)
plt.title('Gender Distribution')
plt.show()

# FirstGen
print(f'\nFirstGen: \n{df["demo_firstgen"].value_counts()}')
print(f'\nFirstGen: \n{df["demo_firstgen"].value_counts(normalize=True)*100}')
sns.countplot(x='demo_firstgen', data=df)
plt.title('FirstGen Distribution')
plt.show()

# Race
print(f'\nRace: \n{df["demo_race"].value_counts()}')
print(f'\nRace: \n{df["demo_race"].value_counts(normalize=True)*100}')
sns.countplot(x='demo_race', data=df)
plt.title('Race Distribution')
plt.show()

# Explore distributions of each categorical column (demo_gender, demo_race, demo_firstgen, cohort)

# Create histogram for each categorical variable

categorical_columns = ['demo_race', 'demo_gender', 'demo_firstgen', 'cohort']
num_rows = 2  # Number of rows for the subplots
num_cols = 2  # Number of columns for the subplots

# Create a figure with subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

# Loop through categorical variables and create histograms on subplots
col_counter = 0
row_counter = 0
for column in categorical_columns:
    ax = axes[row_counter, col_counter]  # Access the current subplot
    sns.histplot(df[column], bins=len(df[column].unique()), ax=ax)
    plt.title(f'Histogram of {column}', fontsize=19)
    plt.xlabel(column, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    col_counter += 1
    if col_counter == num_cols:
        col_counter = 0
        row_counter += 1

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# List categorical variables
categorical_columns = ['demo_race', 'demo_gender', 'demo_firstgen', 'cohort']
num_rows = 2  # Number of rows for the subplots
num_cols = 2  # Number of columns for the subplots

# Create a figure with subplots and adjust spacing
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.88, wspace=0.4, hspace=0.4)  # Adjust margins and spacing

# Loop through categorical variables and create violin plots on subplots
col_counter = 0
row_counter = 0
for column in categorical_columns:
    ax = axes[row_counter, col_counter]  # Access the current subplot
    sns.violinplot(y=df[column], ax=ax)  # Specify x and y for violin plot
    plt.title(f'Violin Plot of {column}', fontsize=19)  # Adjust font size for clarity
    plt.xlabel(column, fontsize=10)
    col_counter += 1
    if col_counter == num_cols:
        col_counter = 0
        row_counter += 1

# Show the plot
plt.show()

# Numerical Variables
# Examine the distribution of numerical variables (e.g., bedtime_mssd, TotalSleepTime, midpoint_sleep, frac_nights_with_data, study, daytime_sleep, cum_gpa, term_gpa) using histograms, box plots, and summary statistics.
# Identify outliers, skewness, and kurtosis.

# List of variables to analyze
variables = ['bedtime_mssd', 'TotalSleepTime', 'midpoint_sleep', 'frac_nights_with_data', 'daytime_sleep', 'study', 'cum_gpa',
             'term_gpa', 'term_units', 'Zterm_units_ZofZ']
num_rows = 5  # Number of rows for the subplots
num_cols = 2  # Number of columns for the subplots

# Create a figure with subplots and adjust spacing
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.88, wspace=0.4, hspace=0.4)  # Adjust margins and spacing

# Loop through variables and create histograms on subplots
row_counter = 0
col_counter = 0
for var in variables:
    ax = axes[row_counter, col_counter]  # Access the current subplot

    # Print descriptive statistics
    print(f'\n{var.upper()}: \n{df[var].describe()}')

    # Create histogram
    sns.histplot(df[var], ax=ax)

    # Set title and labels
    plt.title(f'{var.upper()} Distribution', fontsize=10)
    plt.xlabel(var, fontsize=10)
    plt.ylabel('Count', fontsize=10)

    # Update counters and handle overflow
    col_counter += 1
    if col_counter == num_cols:
        col_counter = 0
        row_counter += 1

plt.tight_layout()

# Show the plot
plt.show()

# Numerical Outliers
# Identify outliers in numerical variables using techniques like Winsorization or trimming.

# List of variables to analyze
variables = ['bedtime_mssd', 'TotalSleepTime', 'midpoint_sleep', 'frac_nights_with_data', 'daytime_sleep', 'study', 'cum_gpa',
             'term_gpa', 'term_units', 'Zterm_units_ZofZ']
num_rows = 2  # Number of rows for the subplots
num_cols = 5  # Number of columns for the subplots

# Create a figure with subplots and adjust spacing
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.88, wspace=0.4, hspace=0.4)  # Adjust margins and spacing

# Loop through variables and create histograms on subplots
row_counter = 0
col_counter = 0
for var in variables:
    ax = axes[row_counter, col_counter]  # Access the current subplot

    # Create Boxplot
    sns.boxplot(df[var], ax=ax)

    # Set title and labels
    plt.title(f'{var.upper()} Distribution', fontsize=10)
    plt.xlabel(var, fontsize=10)
    plt.ylabel('Count', fontsize=10)

    # Update counters and handle overflow
    col_counter += 1
    if col_counter == num_cols:
        col_counter = 0
        row_counter += 1

plt.tight_layout()

# Show the plot
plt.show()


# Winsorization
from scipy.stats.mstats import winsorize

# List all variables
variables = ['bedtime_mssd', 'TotalSleepTime', 'midpoint_sleep', 'frac_nights_with_data', 'daytime_sleep', 'study', 'cum_gpa',
             'term_gpa', 'term_units', 'Zterm_units_ZofZ']

# Winsorize each variable
for var in variables:
    df[var] = winsorize(df[var], limits=[0.05, 0.05])  # Winsorize at the 5th and 95th percentiles
    print(f'\n{var.upper()}: \n{df[var].describe()}')
    print(f'\n{var.upper()}: \n{df[var].value_counts()}')
    print(f'\n{var.upper()}: \n{df[var].value_counts(normalize=True)*100}')
    sns.histplot(df[var])
    plt.title(f'{var.upper()} Distribution', fontsize=10)
    plt.xlabel(var, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.show()
    sns.boxplot(df[var])
    plt.title(f'{var.upper()} Distribution', fontsize=10)
    plt.xlabel(var, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.show()

# Analyze the relationship between categorical variables using cross-tabulation and chi-square tests.
# For example, examine the relationship between demo_race and demo_gender.

from scipy.stats import chi2_contingency
from IPython.display import display

# List categorical variables
categorical_columns = ['demo_race', 'demo_gender', 'demo_firstgen', 'cohort']

for i in range(len(categorical_columns)):
    for j in range(i + 1, len(categorical_columns)):
         var1 = categorical_columns[i]
         var2 = categorical_columns[j]

         # Create cross-tabulation
         cross_tab = pd.crosstab(df[var1], df[var2])

         # Perform chi-square test
         chi2, p_value, _, _ = chi2_contingency(cross_tab)

         print(f"\nCross-tabulation for {var1} and {var2}:")
         display(cross_tab.style.background_gradient(cmap='Blues'))
         print(f"\nChi-square statistic: {chi2:.3f}")
         print(f"P-value: {p_value:.3f}")

         if p_value < 0.05:
                print(f"There is a significant association between {var1} and {var2}.")
         else:
                print(f"There is no significant association between {var1} and {var2}.")


# Calculate correlation coefficients between numerical variables to identify potential relationships.
# For example, assess the correlation between bedtime_mssd and TotalSleepTime.

# List numerical variables
numerical_variables = ['bedtime_mssd', 'TotalSleepTime', 'midpoint_sleep', 'frac_nights_with_data', 'daytime_sleep', 'study', 'cum_gpa',
             'term_gpa', 'term_units', 'Zterm_units_ZofZ']


# Calculate correlation matrix
correlation_matrix = df[numerical_variables].corr()

# Print correlation matrix
print(f'Correlation Matrix: \n{correlation_matrix}')

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=19)
plt.show()

# Explore the relationship between categorical and numerical variables using group statistics, box plots, and ANOVA.
# For example, compare the average TotalSleepTime across different demo_gender categories.

# Import libraries

from statsmodels.formula.api import ols
import statsmodels.api as sm # import the statsmodels API and alias it as 'sm'

# List catergorical and numerical variables

categorical_columns = ['demo_race', 'demo_gender', 'demo_firstgen', 'cohort']
numerical_variables = ['bedtime_mssd', 'TotalSleepTime', 'midpoint_sleep', 'frac_nights_with_data',
                       'daytime_sleep', 'study', 'cum_gpa', 'term_gpa', 'term_units', 'Zterm_units_ZofZ']

for cat_var in categorical_columns:
    for num_var in numerical_variables:
        # Group by categorical variable and calculate mean
        grouped_data = df.groupby(cat_var)[num_var].mean()

        # Print group statistics
        print(f"\nGroup statistics for {num_var} by {cat_var}:")
        print(grouped_data)

        # Create box plot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=cat_var, y=num_var, data=df)
        plt.title(f"{num_var} by {cat_var}")
        plt.xlabel(cat_var)
        plt.ylabel(num_var)
        plt.show()

        # Perform ANOVA
        model = ols(f"{num_var} ~ {cat_var}", data=df).fit()
        anova_table = sm.stats.anova_lm(model)

        print("\nANOVA Table:")
        print(anova_table)

        # Statsitical Significance
        if anova_table['PR(>F)'][0] < 0.05:
            print(f"\nThere is a significant difference in {num_var} between the groups.")
            print(f"\nThe p-value is {anova_table['PR(>F)'][0]}")
        else:
            print(f"\nThere is no significant difference in {num_var} between the groups.")
# Consider techniques like Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE)
# to reduce the dimensionality of the dataset and identify underlying patterns.

# Import libraries

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Separate categorical and numerical variables

categorical_columns = ['cohort', 'demo_race', 'demo_gender', 'demo_firstgen', 'term_units', 'Zterm_units_ZofZ']  # Update if needed
numerical_variables = [col for col in df.columns if col not in categorical_columns]

# Standardize numerical variables for PCA

scaler = StandardScaler()
df[numerical_variables] = scaler.fit_transform(df[numerical_variables])

# Impute NaN values using the mean strategy (replace with preferred strategy)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of the column
df[numerical_variables] = imputer.fit_transform(df[numerical_variables])

# Perform PCA and t-SNE (already included)

pca = PCA(n_components=2)  # Adjust the number of components as needed
pca_results = pca.fit_transform(df[numerical_variables])

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(df[numerical_variables])

# Create subplots for each categorical variable

n_cols = 2  # Number of columns for PCA and t-SNE
n_rows = len(categorical_columns) # Number of rows for each categorical variable

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))  # Adjust figsize as needed

# Loop through categorical variables and create subplots

for i, col in enumerate(categorical_columns):
    # Convert categorical values to numerical labels
    unique_values = df[col].unique()
    mapping = {value: j for j, value in enumerate(unique_values)}
    color_labels = df[col].map(mapping)

    # PCA subplot

    axes[i, 0].scatter(pca_results[:, 0], pca_results[:, 1], c=color_labels)
    axes[i, 0].set_title(f'PCA by {col}')
    axes[i, 0].set_xlabel('Principal Component 1')
    axes[i, 0].set_ylabel('Principal Component 2')

    # t-SNE subplot

    axes[i, 1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=color_labels)
    axes[i, 1].set_title(f't-SNE by {col}')
    axes[i, 1].set_xlabel('t-SNE Component 1')
    axes[i, 1].set_ylabel('t-SNE Component 2')

plt.tight_layout()
plt.show()

# Explore clustering algorithms (e.g., K-means, hierarchical clustering) to group similar participants based on their characteristics.

# Consider techniques like Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE)
# to reduce the dimensionality of the dataset and identify underlying patterns.

# Import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Separate categorical and numerical variables
categorical_columns = ['cohort', 'demo_race', 'demo_gender', 'demo_firstgen', 'term_units', 'Zterm_units_ZofZ']
numerical_variables = [col for col in df.columns if col not in categorical_columns]

# Standardize numerical variables for PCA
scaler = StandardScaler()
df[numerical_variables] = scaler.fit_transform(df[numerical_variables])

# Impute NaN values using the mean strategy
imputer = SimpleImputer(strategy='mean')
df[numerical_variables] = imputer.fit_transform(df[numerical_variables])

# Perform PCA and t-SNE
pca = PCA(n_components=2)
pca_results = pca.fit_transform(df[numerical_variables])

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(df[numerical_variables])

# K-Means Clustering
k_range = range(2, 10)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[numerical_variables])
    labels = kmeans.labels_

    silhouette_score_k = silhouette_score(df[numerical_variables], labels)
    silhouette_scores.append(silhouette_score_k)

best_k = k_range[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(df[numerical_variables])
kmeans_labels = kmeans.labels_

# Hierarchical Clustering
ward_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
ward_cluster.fit(df[numerical_variables])
ward_labels = ward_cluster.labels_

# Visualize Clusters in Subplots
n_cols = 2
n_rows = len(categorical_columns)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))

for i, col in enumerate(categorical_columns):
    # Convert categorical values to numerical labels
    unique_values = df[col].unique()
    mapping = {value: j for j, value in enumerate(unique_values)}
    color_labels = df[col].map(mapping)

    # PCA subplot with cluster labels
    axes[i, 0].scatter(pca_results[:, 0], pca_results[:, 1], c=kmeans_labels)
    axes[i, 0].set_title(f'PCA by {col} (K-Means)')

    # t-SNE subplot with cluster labels
    axes[i, 1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans_labels)
    axes[i, 1].set_title(f't-SNE by {col} (K-Means)')

    # Repeat for hierarchical clustering (replace kmeans_labels with ward_labels)
    axes[i, 0].scatter(pca_results[:, 0], pca_results[:, 1], c=ward_labels)
    axes[i, 0].set_title(f'PCA by {col} (Hierarchical)')

    axes[i, 1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=ward_labels)
    axes[i, 1].set_title(f't-SNE by {col} (Hierarchical)')

plt.tight_layout()
plt.show()

# Use regression models to predict target variables (e.g., term_gpa) based on other variables.

# Import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Separate categorical and numerical variables
categorical_columns = ['cohort', 'demo_race', 'demo_gender', 'demo_firstgen', 'term_units', 'Zterm_units_ZofZ']
numerical_variables = [col for col in df.columns if col not in categorical_columns]

# Target variable (replace 'cum_gpa' with your target variable)
target_variable = 'cum_gpa'

# Standardize numerical variables for PCA
scaler = StandardScaler()
df[numerical_variables] = scaler.fit_transform(df[numerical_variables])

# Impute NaN values using the mean strategy
from sklearn.impute import SimpleImputer # Make sure to import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[numerical_variables] = imputer.fit_transform(df[numerical_variables])

# Define x and y
x = df[numerical_variables]  # Features (independent variables)
y = df[target_variable]    # Target variable

# Linear Regression with L1 regularization (Lasso)
alpha = 0.01  # Adjust alpha for the regularization strength
model = Lasso(alpha=alpha)
model.fit(x, y)

y_pred = model.predict(x)
residuals = y - y_pred

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Feature Importance
perm_importance = permutation_importance(model, x, y, n_repeats=30, random_state=42)

importances = perm_importance.importances_mean
indices = np.argsort(importances)[::-1]

for f in range(x.shape[1]):
    print(f"{x.columns[indices[f]]}: {importances[indices[f]]:.2f}")

# Create subplot grid
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Residual Plot
ax[0].scatter(y_pred, residuals)
ax[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax[0].set_xlabel("Predicted Values")
ax[0].set_ylabel("Residuals")
ax[0].set_title("Residual Plot")
ax[0].grid(True)

# Feature Importance Bar Chart
ax[1].barh(range(x.shape[1]), importances[indices], color='blue')
ax[1].set_yticks(range(x.shape[1]))
ax[1].set_yticklabels(x.columns[indices], rotation=45)
ax[1].set_xlabel('Feature Importance')
ax[1].set_ylabel('Feature')
ax[1].set_title('Permutation Feature Importance')

plt.tight_layout()
plt.show()