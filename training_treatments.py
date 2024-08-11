import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, GridSearchCV, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
from tqdm import tqdm
from scipy.stats import norm, binomtest

# Suppress warnings
warnings.filterwarnings("ignore")


def format_range(low, high):
    if pd.isna(low) or pd.isna(high):
        return "N/A"
    return f"{low:.1f} to {high:.1f}"


def confidence_interval(data, confidence=0.95, min_value=None, max_value=None):
    n = len(data)
    if n < 2:
        return data.iloc[0], data.iloc[0]
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    t_value = norm.ppf((1 + confidence) / 2)
    margin_error = t_value * std_err
    low = mean - margin_error
    high = mean + margin_error
    if min_value is not None:
        low = max(min_value, low)
    if max_value is not None:
        high = min(max_value, high)
    return low, high


def format_p_value(p):
    return '<0.001' if p < 0.001 else f'{p:.4f}'


# Load the CSV file
df = pd.read_csv('data/MRI_NFL_cogn_type_filter - stage2.csv')

# Remove treatments with fewer than 10 samples
treatment_counts = df['Treatment'].value_counts()
valid_treatments = treatment_counts[treatment_counts >= 10].index
df = df[df['Treatment'].isin(valid_treatments)]

print(f"Treatments with at least 10 samples: {', '.join(valid_treatments)}")


# Calculate significant treatments
def get_treatment_summary(group):
    proportion = (group['Treatment Response'] == 'Yes').mean()
    sample_size = len(group)
    successes = (group['Treatment Response'] == 'Yes').sum()
    p_value = binomtest(successes, sample_size, p=0.5).pvalue
    age_ci = confidence_interval(group['age'])
    edss_ci = confidence_interval(group['EDSS'])
    return pd.Series({
        'Proportion': proportion,
        'Patients': sample_size,
        'Age': format_range(*age_ci),
        'EDSS': format_range(*edss_ci),
        'MS-Type': group['MS Type'].mode().iloc[0],
        'p-value': p_value
    })


treatment_effectiveness = df.groupby('Treatment').apply(get_treatment_summary).reset_index()
significant_treatments = treatment_effectiveness[(treatment_effectiveness['p-value'] < 0.05) &
                                                 (treatment_effectiveness['Proportion'] >= 0.5)]['Treatment'].tolist()

# Filter the dataframe to include only significant treatments
df_significant = df[df['Treatment'].isin(significant_treatments)]

# Analyze proportion for significant treatments
print("\nProportion by Significant Treatment:")
treatment_proportion = df_significant.groupby('Treatment')['Treatment Response'].apply(
    lambda x: (x == 'Yes').mean()).sort_values(
    ascending=False)
print(treatment_proportion)

# Analyze treatment effectiveness for significant treatments
print("\nTreatment Effectiveness Analysis (Significant Treatments):")
for treatment in significant_treatments:
    treatment_data = df_significant[df_significant['Treatment'] == treatment]
    print(f"\nTreatment: {treatment}")
    print(f"Overall Proportion: {(treatment_data['Treatment Response'] == 'Yes').mean():.2f}")
    age_ci = confidence_interval(treatment_data['age'])
    edss_ci = confidence_interval(treatment_data['EDSS'])
    print(f"Age: {format_range(*age_ci)}")
    print(f"EDSS: {format_range(*edss_ci)}")
    print("MS Type Distribution:")
    print(treatment_data['MS Type'].value_counts(normalize=True))

# Visualizations for significant treatments
plt.figure(figsize=(12, 6), dpi=400)
sns.boxplot(x='Treatment', y='age', hue='Treatment Response', data=df_significant)
plt.title('Treatment Effectiveness by Age (Significant Treatments)')
plt.xticks(rotation=45, ha='right')
plt.ylim(17, 80)
plt.tight_layout()
plt.savefig('treatment_age_response_significant.png', dpi=400)
plt.close()

plt.figure(figsize=(12, 6), dpi=400)
sns.boxplot(x='Treatment', y='EDSS', hue='Treatment Response', data=df_significant)
plt.title('Treatment Effectiveness by EDSS (Significant Treatments)')
plt.xticks(rotation=45, ha='right')
plt.ylim(1, 8)
plt.tight_layout()
plt.savefig('treatment_edss_response_significant.png', dpi=400)
plt.close()

# Heatmap of proportion by Treatment and MS Type (for significant treatments only)
proportion_heatmap = df_significant.pivot_table(values='Treatment Response', index='Treatment', columns='MS Type',
                                                aggfunc=lambda x: (x == 'Yes').mean())
plt.figure(figsize=(10, 8), dpi=400)
sns.heatmap(proportion_heatmap, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Proportion by Treatment and MS Type (Significant Treatments)')
plt.tight_layout()
plt.savefig('treatment_ms_type_heatmap_significant.png', dpi=400)
plt.close()

# Create a table of treatment effectiveness for significant treatments
significant_combinations = treatment_effectiveness[treatment_effectiveness['Treatment'].isin(significant_treatments)]
best_combinations = significant_combinations.sort_values('Proportion', ascending=False)

# Format p-values for display
best_combinations['p-value'] = best_combinations['p-value'].apply(format_p_value)

# Display the table in the console
print("\nBest Treatment Combinations (p < 0.05 and Proportion >= 0.5):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
print(best_combinations)

# Save the table to a CSV file
best_combinations.to_csv('best_treatment_combinations_significant.csv', index=False)

# Model Selection and Grid Search
X = df_significant[['age', 'MS Type', 'EDSS', 'Lesion', 'WB']].copy()
y = df_significant['Treatment Response'].map({'No': 0, 'Yes': 1})

# Encode categorical variables
le_ms_type = LabelEncoder()
X['MS Type'] = le_ms_type.fit_transform(X['MS Type'])

# Define reduced parameter grids for Grid Search
dt_param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

gb_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'subsample': [1.0]
}

rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

svm_param_grid = {
    'svm__C': [0.1, 1],
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': ['scale', 'auto']
}

knn_param_grid = {
    'n_neighbors': [3, 5],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


# Function to perform GridSearchCV with progress bar
def grid_search_model(model, param_grid, X, y, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=6, scoring='f1_weighted', n_jobs=-1)

    with tqdm(total=1, desc=f"GridSearch for {model_name}", unit="fit") as pbar:
        grid_search.fit(X, y)
        pbar.update(1)

    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_


# Generate classification reports for each model
def get_classification_report(model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=6, n_jobs=-1)
    report = classification_report(y, y_pred, target_names=['No', 'Yes'])
    f1 = f1_score(y, y_pred, average='weighted')
    return report, f1


# Perform grid search and generate classification reports
print("Starting Grid Search and generating classification reports for all models...")

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
best_dt, dt_score, dt_params = grid_search_model(dt_model, dt_param_grid, X, y, "Decision Tree")
dt_report, dt_f1 = get_classification_report(best_dt, X, y)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
best_gb, gb_score, gb_params = grid_search_model(gb_model, gb_param_grid, X, y, "Gradient Boosting")
gb_report, gb_f1 = get_classification_report(best_gb, X, y)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
best_rf, rf_score, rf_params = grid_search_model(rf_model, rf_param_grid, X, y, "Random Forest")
rf_report, rf_f1 = get_classification_report(best_rf, X, y)

# SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42, probability=True))
])
best_svm, svm_score, svm_params = grid_search_model(svm_pipeline, svm_param_grid, X, y, "SVM")
svm_report, svm_f1 = get_classification_report(best_svm, X, y)

# KNN
knn_model = KNeighborsClassifier()
best_knn, knn_score, knn_params = grid_search_model(knn_model, knn_param_grid, X, y, "KNN")
knn_report, knn_f1 = get_classification_report(best_knn, X, y)

# Print classification reports
print("\nDecision Tree Classification Report:")
print(dt_report)
print("Best parameters:", dt_params)

print("\nGradient Boosting Classification Report:")
print(gb_report)
print("Best parameters:", gb_params)

print("\nRandom Forest Classification Report:")
print(rf_report)
print("Best parameters:", rf_params)

print("\nSVM Classification Report:")
print(svm_report)
print("Best parameters:", svm_params)

print("\nKNN Classification Report:")
print(knn_report)
print("Best parameters:", knn_params)

# Determine the best model
models = {
    "Decision Tree": (best_dt, dt_f1, dt_params),
    "Gradient Boosting": (best_gb, gb_f1, gb_params),
    "Random Forest": (best_rf, rf_f1, rf_params),
    "SVM": (best_svm, svm_f1, svm_params),
    "KNN": (best_knn, knn_f1, knn_params)
}

best_model_name = max(models, key=lambda k: models[k][1])
best_model, best_score, best_params = models[best_model_name]

print(f"\nBest overall model: {best_model_name}")
print(f"Best F1 Score: {best_score}")
print(f"Optimal hyperparameters: {best_params}")

# Save the best model
joblib.dump(best_model, 'ms_treatment_model.joblib')
print(f"Best model saved as 'ms_treatment_model.joblib'")

# Feature importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_imp)

    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title(f'Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=300)
    plt.close()
    print("Feature importance graph saved as 'feature_importances.png'")
elif best_model_name == "Logistic Regression":
    # For Logistic Regression, we can use the coefficients as feature importance
    importances = abs(best_model.coef_[0])
    feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)

    print("\nFeature Importances (absolute values of coefficients):")
    print(feature_imp)

    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title(f'Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=300)
    plt.close()
    print("Feature importance graph saved as 'feature_importances.png'")
else:
    print("\nFeature importances not available for this model type.")

print(
    "\nAnalysis complete. Check the generated PNG files for visualizations and 'best_treatment_combinations_significant.csv' for the table of significant treatment combinations (p < 0.05 and Proportion >= 0.5).")


# Calculate and plot response probabilities for significant treatments
def calculate_ci(successes, total, confidence=0.95):
    p = successes / total
    se = np.sqrt(p * (1 - p) / total)
    z = norm.ppf((1 + confidence) / 2)
    margin = z * se
    return p - margin, p + margin


# Calculate proportions and confidence intervals
treatment_stats = df_significant.groupby('Treatment').agg({
    'Treatment Response': lambda x: (x == 'Yes').mean(),
    'Treatment': 'count'
}).rename(columns={'Treatment': 'Count'}).reset_index()

treatment_stats['Lower CI'], treatment_stats['Upper CI'] = zip(*treatment_stats.apply(
    lambda row: calculate_ci((row['Treatment Response'] * row['Count']), row['Count']), axis=1
))

# Sort treatments by response proportion
treatment_stats = treatment_stats.sort_values('Treatment Response', ascending=False)

# Create the plot
plt.figure(figsize=(12, 6))
sns.pointplot(x='Treatment', y='Treatment Response', data=treatment_stats,
              join=False, color='black', scale=0.7)

# Add error bars
plt.errorbar(x=range(len(treatment_stats)),
             y=treatment_stats['Treatment Response'],
             yerr=[treatment_stats['Treatment Response'] - treatment_stats['Lower CI'],
                   treatment_stats['Upper CI'] - treatment_stats['Treatment Response']],
             fmt='none', color='black', capsize=3)

# Customize the plot
plt.title('Response Probability for Significant Treatments', fontsize=16)
plt.xlabel('Treatment', fontsize=12)
plt.ylabel('P(Treatment Response = Yes)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)  # Set y-axis limits
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add reference line at 0.5
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('significant_treatments_response_probability.png', dpi=300)
plt.close()

print(
    "Graph of response probabilities for significant treatments has been saved as 'significant_treatments_response_probability.png'")

# Generate the learning curve for the best model
if best_model_name in ["Decision Tree", "Gradient Boosting", "Random Forest"]:
    param_range = np.arange(1, 21)
    train_scores, test_scores = validation_curve(
        best_model, X, y, param_name="max_depth", param_range=param_range,
        cv=6, scoring="f1_weighted", n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_mean, label="Training score", color="r")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="g")

    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="r", alpha=0.2)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="g", alpha=0.2)

    plt.title(f"Validation Curve with {best_model_name}")
    plt.xlabel("Max Depth")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig('validation_curve.png', dpi=300)
    plt.close()
    print("\nLearning curve has been saved as 'validation_curve.png'")
else:
    print("\nValidation curve generation is not supported for this model type.")

print("\nAnalysis and visualization complete.")
