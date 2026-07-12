# NeuroSage

## Alzheimer's Disease Prediction using Machine Learning

NeuroSage is a machine learning project that demonstrates a complete supervised learning workflow using an Alzheimer's disease dataset. The project covers data acquisition, cleaning, exploratory data analysis (EDA), preprocessing, regression modeling, binary classification, model evaluation, and performance comparison.

This project was developed as part of a supervised machine learning capstone project.

---

# Project Objectives

The objectives of this project are to:

- Perform data cleaning and exploratory data analysis.
- Identify missing values, duplicates, skewness, and outliers.
- Visualize important dataset characteristics.
- Build regression and classification models.
- Evaluate model performance using appropriate statistical metrics.
- Compare regularized and non-regularized machine learning models.
- Demonstrate best practices for preprocessing and avoiding data leakage.

---

# Dataset

The project uses an Alzheimer's Disease clinical dataset containing patient demographic, lifestyle, cardiovascular, cognitive, and clinical variables.

Dataset summary:

- Total observations: **2149**
- Total variables: **35**

Example features include:

- Age
- Gender
- BMI
- Smoking
- Physical Activity
- Diet Quality
- Sleep Quality
- Blood Pressure
- Cholesterol
- Functional Assessment
- Memory Complaints
- Alzheimer's Diagnosis

---

# Repository Structure

```
NeuroSage/

├── data/
├── notebooks/
│   ├── Part1_Data_Cleaning_EDA.ipynb
│   └── Part2_Supervised_Machine_Learning.ipynb
│
├── outputs/
│   └── cleaned_data.csv
│
├── figures/
├── app/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis

## Data Inspection

The dataset was loaded using pandas and inspected using:

- Head of dataset
- Dataset shape
- Data types
- Column names

---

## Missing Value Analysis

Missing values were calculated for every column.

Results:

- No missing values were found.
- Therefore no imputation was required.

Although no imputation was necessary, the **median** would be preferred over the mean for skewed variables because the median is less affected by extreme values.

---

## Duplicate Analysis

Duplicate rows were checked using:

```
df.duplicated().sum()
```

Result:

- No duplicate rows were found.

---

## Data Type Optimization

The repetitive binary variables were converted into efficient numerical representations where appropriate.

Memory usage was reduced from approximately:

- Before optimization: **709302 bytes**
- After optimization: **438848 bytes**

---

## Descriptive Statistics

Summary statistics were generated using:

```
df.describe()
```

This included:

- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

---

## Skewness Analysis

Skewness was computed for all numerical variables.

The highest absolute skewness was observed for:

**SleepQuality**

Skewness value:

```
-0.0696
```

This indicates that the variable is approximately symmetric.

---

## Outlier Detection

Outliers were detected using the Interquartile Range (IQR) method.

Columns analysed:

- BMI
- MMSE

Results:

- No statistical outliers were detected.
- Therefore all observations were retained.

---

## Exploratory Data Analysis

The following visualizations were created:

- Line Plot
- Bar Chart
- Histogram
- Scatter Plot
- Box Plot
- Correlation Heatmap

The visualizations were used to understand feature distributions and relationships prior to machine learning.

---

## Correlation Analysis

Both Pearson and Spearman correlation matrices were computed.

The differences between Pearson and Spearman correlations were very small, indicating that most relationships in the dataset were approximately linear.

Therefore Pearson correlation was selected as the primary guide for feature interpretation.

---

## Grouped Aggregation

Grouped aggregation was performed using:

- Gender
- Age

Results showed only minimal differences between groups, suggesting that Gender alone provides limited predictive information for age-related variation.

---

# Part 2 – Supervised Machine Learning

## Feature Selection

Regression Target

```
MMSE
```

Classification Target

```
Diagnosis
```

Predictor variables consisted of all remaining clinical features after removing:

- PatientID
- DoctorInCharge
- Target variables

---

## Data Preprocessing

The dataset was divided into:

- Training Set (80%)
- Test Set (20%)

Feature scaling was performed using StandardScaler.

To avoid **data leakage**, the scaler was fitted only on the training data and then applied to the testing data.

---

# Regression Model

## Linear Regression

Performance:

- Mean Squared Error (MSE): **73.509**
- R² Score: **-0.0157**

Top three coefficients:

- Diabetes
- Diastolic Blood Pressure
- Cardiovascular Disease

Positive coefficients indicate an increase in predicted MMSE score as the standardized feature increases, whereas negative coefficients indicate a decrease in predicted MMSE score.

---

## Ridge Regression

Performance:

- Mean Squared Error (MSE): **73.508**
- R² Score: **-0.0156**

Ridge Regression produced performance very similar to Linear Regression.

The Ridge parameter (**alpha**) controls the amount of L2 regularization applied to the regression coefficients.

---

# Classification Model

A Logistic Regression classifier was trained to predict Alzheimer's Diagnosis.

Training class distribution:

- Class 0: 64.69%
- Class 1: 35.31%

Since the minority class exceeded 35%, no additional balancing techniques (such as SMOTE or class weighting) were required.

---

## Model Evaluation

Evaluation included:

- Confusion Matrix
- Classification Report
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Area Under the ROC Curve (AUC)

The model achieved:

**AUC = 0.864**

This indicates good ability to distinguish between Alzheimer's positive and negative patients.

---

## Precision and Recall

Precision

```
TP / (TP + FP)
```

Recall

```
TP / (TP + FN)
```

For Alzheimer's disease prediction, **Recall** is particularly important because failing to identify an affected patient (false negative) may delay diagnosis and treatment.

---

## Decision Threshold Analysis

Thresholds evaluated:

- 0.30
- 0.40
- 0.50
- 0.60
- 0.70

The highest F1-score was obtained at:

```
Threshold = 0.30
```

Lowering the threshold improves recall at the cost of increased false positives.

---

## Logistic Regression Regularization

A second Logistic Regression model was trained using:

```
C = 0.01
```

Results were compared against the baseline model (C = 1.0).

Reducing C increases regularization strength by shrinking model coefficients.

Performance differences between the two models were minimal.

---

## Bootstrap Confidence Interval

Bootstrap resampling (500 iterations) was performed to compare the AUC difference between the two Logistic Regression models.

Results:

- Mean Difference: **-0.00215**
- 95% Confidence Interval:
  - Lower: **-0.00444**
  - Upper: **0.000012**

Because the confidence interval includes zero, the observed difference between the two models is not statistically reliable.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# Future Improvements

Potential future work includes:

- Random Forest
- XGBoost
- Support Vector Machines
- Hyperparameter Optimization
- Explainable AI using SHAP
- Deep Learning models for Alzheimer's prediction

---

# Author

**Rethi Shaliya**

Project: **NeuroSage**
