# NeuroSage

## Alzheimer's Disease Prediction using Machine Learning

---

## Project Overview

NeuroSage is a machine learning project developed to predict Alzheimer's disease using clinical and demographic data. The project demonstrates a complete machine learning workflow beginning with raw data preprocessing, exploratory data analysis (EDA), supervised learning, ensemble learning, hyperparameter tuning, and model deployment.

The project was developed using Python and Scikit-learn and follows good machine learning practices such as data cleaning, prevention of data leakage, model evaluation, cross-validation, and pipeline construction.

---

# Objectives

- Clean and preprocess a real-world Alzheimer's dataset.
- Explore relationships between clinical variables.
- Build regression and classification models.
- Compare multiple machine learning algorithms.
- Tune model hyperparameters.
- Evaluate models using appropriate performance metrics.
- Save the best trained model for future prediction.

---

# Dataset

The dataset contains demographic, lifestyle, medical history, cognitive assessment, and laboratory measurements collected from Alzheimer's disease patients.

Target Variable:

- Diagnosis
    - 0 = No Alzheimer's Disease
    - 1 = Alzheimer's Disease

Number of samples:

- 2149

Features include:

- Age
- Gender
- BMI
- Smoking
- Alcohol Consumption
- Physical Activity
- Sleep Quality
- Blood Pressure
- Cholesterol
- MMSE
- Functional Assessment
- Memory Complaints
- ADL
- Behavioural Problems
- and several additional clinical variables.

---

# Project Structure

```
NeuroSage
│
├── data
│   └── alzheimers.csv
│
├── notebooks
│   ├── Part1_Data_Cleaning_EDA.ipynb
│   ├── Part2_Supervised_Machine_Learning.ipynb
│   └── Part3_Advanced_Modeling.ipynb
│
├── outputs
│   └── cleaned_data.csv
│
├── models
│   └── best_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---
# Part 1 — Data Acquisition, Cleaning, and Exploratory Data Analysis

## Data Loading

The Alzheimer's disease dataset was loaded into a Pandas DataFrame using `pd.read_csv()`. The dataset contained **2,149 rows** and **35 columns** before preprocessing.

The first five records, column names, data types, and dataset dimensions were inspected to understand the structure of the dataset.

---

## Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
(df.isnull().sum()/df.shape[0])*100
```

Columns with missing values below 20% were imputed using the **median**.

### Why Median Instead of Mean?

The median is less affected by extreme values (outliers) than the mean. Several clinical variables showed skewed distributions; therefore, median imputation preserved the central tendency more accurately.

After imputation, all remaining missing values were removed successfully.

---

## Duplicate Detection

Duplicate rows were identified using:

```python
df.duplicated().sum()
```

Duplicates were removed using:

```python
df.drop_duplicates()
```

After duplicate removal, the dataset remained clean and no significant change in missing-value percentages was observed.

---

## Data Type Verification

Data types of all columns were inspected.

The categorical administrative column (`DoctorInCharge`) was removed because it contained no predictive information.

Numeric columns were maintained as integer or floating-point values for machine learning.

---

## Memory Usage

Memory usage before and after preprocessing was evaluated using:

```python
df.memory_usage(deep=True).sum()
```

Removing unnecessary columns reduced memory consumption and improved computational efficiency.

---

## Descriptive Statistics

Summary statistics were generated using:

```python
df.describe()
```

The analysis included:

- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

These statistics provided an overview of the distribution of each clinical variable.

---

## Skewness Analysis

Skewness was computed for every numeric feature.

The most skewed variables identified were:

- SleepQuality
- DiastolicBP

Both variables exhibited slight negative skewness.

Since negatively skewed distributions have means that are influenced by lower extreme values, the **median** was selected as the preferred statistic for missing-value imputation.

---

## Outlier Detection

Outliers were detected using the Interquartile Range (IQR) method.

The following variables were analysed:

- BMI
- MMSE

For each variable:

- First Quartile (Q1)
- Third Quartile (Q3)
- IQR
- Lower Bound
- Upper Bound

were calculated.

No significant outliers were detected in either BMI or MMSE.

Therefore, no observations were removed. All samples were retained for modelling to preserve dataset size.

---

## Exploratory Visualizations

Five visualization techniques were created:

### Line Plot

A line plot was used to visualize the trend of a continuous variable across observations.

### Bar Chart

A bar chart compared the average value of a numeric variable across categorical groups.

### Histogram

A histogram illustrated the distribution of the most skewed feature.

### Scatter Plot

A scatter plot examined relationships between two continuous variables.

### Box Plot

A box plot compared the spread and median of a numeric feature across categorical groups.

These visualizations provided insight into the overall structure and variability of the dataset.

---

## Correlation Analysis

Pearson correlation coefficients were calculated using:

```python
df.corr()
```

A correlation heatmap was generated to visualize relationships among numeric variables.

To investigate potential monotonic relationships, Spearman rank correlation was also computed using:

```python
df.corr(method="spearman")
```

The absolute differences between Pearson and Spearman correlations were very small, indicating that relationships among the variables were predominantly linear rather than strongly nonlinear.

---

## Grouped Aggregation

Grouped descriptive statistics were calculated using:

```python
df.groupby(categorical_column).agg(["mean","std","count"])
```

The analysis identified:

- Group with the highest mean
- Group with the highest standard deviation
- Ratio between highest and lowest group means

The observed ratio suggested only a modest predictive signal from the selected categorical feature.

---

## Clean Dataset

After preprocessing, the cleaned dataset was exported as:

```
outputs/cleaned_data.csv
```

This cleaned dataset served as the input for all machine learning models developed in Parts 2 and 3.
# Part 2 — Supervised Machine Learning

## Data Preprocessing

The cleaned dataset produced in Part 1 was loaded into a Pandas DataFrame.

### Feature Matrix

The feature matrix **X** consisted of all predictor variables except the target variable.

### Regression Target

The continuous variable **MMSE** (Mini Mental State Examination score) was selected as the regression target.

### Classification Target

The Alzheimer's **Diagnosis** column was used as the binary classification target.

- 0 = No Alzheimer's Disease
- 1 = Alzheimer's Disease

The administrative column **DoctorInCharge** was removed because it contains no predictive information.

---

## Train-Test Split

The dataset was divided into training and testing sets using:

- Training Set: 80%
- Testing Set: 20%
- Random State = 42

To prevent **data leakage**, the StandardScaler was fitted **only on the training dataset** and then applied to both the training and testing data.

If scaling were performed before splitting, information from the testing dataset would leak into model training and produce overly optimistic performance estimates.

---

# Linear Regression

A Linear Regression model was trained using the scaled training data.

## Results

| Metric | Value |
|---------|---------:|
| Mean Squared Error (MSE) | 73.509 |
| R² Score | -0.0157 |

The negative R² value indicates that Linear Regression performs poorly for predicting MMSE in this dataset and is unable to explain the variation in the target variable.

---

## Most Influential Features

The three largest absolute regression coefficients were:

| Feature | Coefficient |
|---------|-----------:|
| Diabetes | -0.339 |
| DiastolicBP | -0.323 |
| CardiovascularDisease | 0.290 |

A positive coefficient indicates that increasing the standardized feature increases the predicted MMSE value, while a negative coefficient indicates the opposite.

---

# Ridge Regression

Ridge Regression was trained using:

```python
Ridge(alpha=1.0)
```

## Results

| Model | MSE | R² |
|---------|---------:|---------:|
| Linear Regression | 73.509 | -0.0157 |
| Ridge Regression | 73.508 | -0.0156 |

The Ridge Regression model produced nearly identical performance to ordinary least squares Linear Regression.

The alpha parameter controls the strength of L2 regularization. Increasing alpha shrinks coefficient values and reduces model variance, helping prevent overfitting.

---

# Logistic Regression

A Logistic Regression classifier was developed to predict Alzheimer's diagnosis.

## Class Distribution

Training data contained:

| Diagnosis | Percentage |
|------------|-----------:|
| No Alzheimer's | 64.69% |
| Alzheimer's | 35.31% |

Since the minority class exceeded 35% of the training data, no class balancing technique was required.

---

## Classification Results

### Confusion Matrix

| | Predicted 0 | Predicted 1 |
|----|-----------:|-----------:|
| Actual 0 | 245 | 32 |
| Actual 1 | 57 | 96 |

---

### Performance Metrics

| Metric | Value |
|---------|---------:|
| Accuracy | 0.793 |
| Precision | 0.750 |
| Recall | 0.627 |
| F1 Score | 0.683 |

Precision measures the proportion of predicted Alzheimer's cases that were actually Alzheimer's.

Recall measures the proportion of actual Alzheimer's patients correctly identified by the model.

For medical diagnosis, recall is generally considered more important because failing to identify a patient (false negative) may delay treatment.

---

# ROC Curve

The Receiver Operating Characteristic (ROC) curve was generated using predicted probabilities.

## Results

| Metric | Value |
|---------|---------:|
| AUC | **0.864** |

An AUC of 0.864 indicates that the classifier has good ability to distinguish Alzheimer's patients from healthy individuals.

---

# Decision Threshold Analysis

Predicted probabilities were evaluated using thresholds from **0.30 to 0.70**.

| Threshold | Precision | Recall | F1 |
|-----------:|----------:|-------:|------:|
| 0.30 | 0.668 | 0.856 | **0.751** |
| 0.40 | 0.734 | 0.739 | 0.736 |
| 0.50 | 0.750 | 0.627 | 0.683 |
| 0.60 | 0.743 | 0.529 | 0.618 |
| 0.70 | 0.758 | 0.451 | 0.566 |

The highest F1-score occurred at a threshold of **0.30**, indicating a better balance between precision and recall than the default threshold of 0.50.

Lowering the decision threshold increases recall but also increases false positives.

---

# Regularization Experiment

A second Logistic Regression model was trained using stronger regularization:

```python
C = 0.01
```

## Results

| Model | Precision | Recall | AUC |
|---------|---------:|---------:|---------:|
| C = 1.0 | 0.750 | 0.627 | 0.864 |
| C = 0.01 | 0.746 | 0.575 | 0.866 |

The parameter **C** controls the inverse strength of regularization.

Lower values of C apply stronger regularization, shrinking model coefficients and reducing model complexity.

---

# Bootstrap Confidence Interval

Bootstrap resampling (500 iterations) was performed to compare the AUC of the two Logistic Regression models.

## Results

| Statistic | Value |
|-----------|---------:|
| Mean AUC Difference | -0.00215 |
| 2.5 Percentile | -0.00444 |
| 97.5 Percentile | 0.000012 |

Because the 95% confidence interval includes zero, the difference between the two Logistic Regression models is not statistically reliable.

Both models demonstrate very similar classification performance.
# Part 3 — Advanced Modeling, Ensemble Learning, and Model Deployment

## Decision Tree Classifier

A baseline Decision Tree classifier was trained using the default hyperparameters.

### Results

| Model | Training Accuracy | Testing Accuracy |
|--------|------------------:|-----------------:|
| Default Decision Tree | **1.000** | **0.900** |

The training accuracy reached 100%, while the testing accuracy was lower (90%), indicating that the unconstrained decision tree overfitted the training data. Decision Trees are considered high-variance models because they greedily split the data to maximize purity at each node, which can result in memorizing training examples.

---

## Controlled Decision Tree

A second Decision Tree was trained using:

- max_depth = 5
- min_samples_split = 20

### Results

| Training Accuracy | Testing Accuracy |
|------------------:|-----------------:|
| **0.961** | **0.930** |

Restricting the maximum tree depth reduced overfitting and improved generalization performance.

- **max_depth** limits how deep the tree can grow, reducing variance.
- **min_samples_split** prevents the tree from creating splits based on very small groups of observations, reducing sensitivity to noise.

---

## Gini vs Entropy

Two Decision Tree models were compared.

| Criterion | Test Accuracy |
|-----------|--------------:|
| Gini | **0.923** |
| Entropy | **0.947** |

### Gini Impurity

\[
Gini = 1-\sum p_i^2
\]

### Entropy

\[
Entropy=-\sum p_i\log_2(p_i)
\]

A node with **Gini = 0** is perfectly pure, meaning every sample in that node belongs to the same class.

---

# Random Forest

A Random Forest classifier was trained using:

- n_estimators = 100
- max_depth = 10
- random_state = 42

### Results

| Metric | Value |
|---------|-------:|
| Training Accuracy | **0.988** |
| Testing Accuracy | **0.949** |
| ROC-AUC | **0.940** |

### Top Five Important Features

| Rank | Feature | Importance |
|------:|---------|-----------:|
| 1 | FunctionalAssessment | 0.194 |
| 2 | ADL | 0.165 |
| 3 | MMSE | 0.137 |
| 4 | MemoryComplaints | 0.096 |
| 5 | BehavioralProblems | 0.047 |

Random Forest feature importance measures the average reduction in Gini impurity contributed by each feature across all trees in the ensemble. Unlike linear regression coefficients, feature importance values do not indicate whether a feature increases or decreases the prediction; instead, they quantify how useful the feature is for making accurate splits.

---

## Bagging Concept

Random Forest uses **Bootstrap Aggregating (Bagging)**.

Each decision tree is trained on a random sample of the training data generated by sampling with replacement. Additionally, only a random subset of predictor variables is considered at each split.

These two sources of randomness reduce model variance and improve robustness compared with a single Decision Tree.

---

# Gradient Boosting

Gradient Boosting was trained using:

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3

### Results

| Metric | Value |
|---------|-------:|
| Training Accuracy | **0.967** |
| Testing Accuracy | **0.947** |
| ROC-AUC | **0.946** |

Gradient Boosting sequentially trains trees so that each new tree focuses on correcting the errors made by the previous ensemble, producing a strong predictive model.

---

# Feature Ablation Study

The five least important Random Forest features were:

- HeadInjury
- PersonalityChanges
- Diabetes
- Depression
- Confusion

### ROC-AUC Comparison

| Model | ROC-AUC |
|---------|--------:|
| Full Random Forest | **0.9463** |
| Reduced Random Forest | **0.9419** |

Removing the least important features caused only a small decrease in ROC-AUC, indicating that these variables contributed limited predictive information. A reduced model may therefore be attractive in production when lower computational cost is more important than a small decrease in predictive performance.

---

# Five-Fold Cross Validation

A Stratified 5-fold cross-validation procedure was used to estimate model generalization performance.

### Results

| Model | Mean AUC | Standard Deviation |
|---------|---------:|------------------:|
| Logistic Regression | **0.894** | 0.0116 |
| Decision Tree | **0.936** | 0.0189 |
| Random Forest | **0.953** | 0.0066 |
| Gradient Boosting | **0.951** | 0.0098 |

Cross-validation provides a more reliable estimate of performance because every observation is used for both training and validation across multiple folds, reducing dependence on a single train-test split.

---

# Hyperparameter Optimization

GridSearchCV was performed using the following search space:

- n_estimators = 50, 100, 200
- max_depth = 5, 10, None
- min_samples_leaf = 1, 5

A total of:

**18 parameter combinations × 5 folds = 90 model fits**

were evaluated.

### Best Parameters

- n_estimators = **200**
- max_depth = **None**
- min_samples_leaf = **1**

### Best Cross-Validated AUC

**0.9562**

Grid Search exhaustively evaluates every parameter combination, whereas Randomized Search evaluates only a random subset, making it computationally faster but potentially less optimal.

---

# Learning Curve

| Training Fraction | Training AUC | Test AUC |
|------------------:|-------------:|---------:|
| 20% | 1.000 | 0.940 |
| 40% | 1.000 | 0.942 |
| 60% | 1.000 | 0.939 |
| 80% | 1.000 | 0.943 |
| 100% | 1.000 | 0.940 |

The model achieved perfect performance on the training data regardless of training fraction, indicating high capacity. Test AUC remained stable around 0.94, suggesting that collecting substantially more data alone is unlikely to produce major improvements. The model appears to have reached a performance plateau.

---

# Model Serialization

The best-performing pipeline was saved using:

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was successfully reloaded using:

```python
loaded_model = joblib.load("best_model.pkl")
```

Predictions on two sample observations were successfully generated, confirming that model serialization and deserialization worked correctly.

---

# Final Model Comparison

| Model | Test AUC / CV AUC |
|---------|------------------:|
| Logistic Regression | 0.864 |
| Random Forest | 0.940 |
| Gradient Boosting | 0.946 |
| Random Forest (5-fold CV) | 0.953 |
| Best GridSearchCV Pipeline | **0.956** |

---

# Final Recommendation

Among all evaluated models, the tuned Random Forest pipeline achieved the highest cross-validated ROC-AUC (0.956). This model demonstrated strong predictive performance, low variability across folds, and excellent generalization. For deployment in a clinical decision-support setting, the tuned Random Forest pipeline is recommended because it provides the best balance between predictive accuracy, robustness, and stability.

---

# Future Improvements

Future work may include:

- XGBoost
- LightGBM
- CatBoost
- Deep Neural Networks
- Explainable AI (SHAP and LIME)
- External validation using independent Alzheimer's datasets
- Hyperparameter optimization using Bayesian optimization

---

# Author

**Rethi Shaliya**

**Project:** NeuroSage – Alzheimer's Disease Prediction using Machine Learning

---

# License

This project was developed for educational and research purposes.
