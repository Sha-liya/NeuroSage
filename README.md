# NeuroSage

## Alzheimer's Disease Prediction Using Machine Learning

## Project Description
This project performs data acquisition, cleaning, exploratory data analysis (EDA), and preparation of an Alzheimer's disease dataset for machine learning.

## Dataset
- Alzheimer's Disease Dataset
- Total records: 2149
- Total features: 35

## Part 1 Completed

### Data Cleaning
- Loaded dataset using pandas
- Checked dataset shape and data types
- Identified missing values
- No missing values were found
- Checked duplicate records
- No duplicate records were found
- Converted repetitive binary variables to category data type
- Reduced memory usage

### Descriptive Statistics
- Generated descriptive statistics using df.describe()
- Calculated skewness for all numerical variables
- SleepQuality showed the highest absolute skewness (-0.0696), indicating an almost symmetric distribution.

### Outlier Analysis
- IQR analysis was performed on BMI and MMSE.
- No outliers were detected.
- No rows were removed.

### Exploratory Data Analysis
The following visualizations were created:
- Line Plot
- Bar Chart
- Histogram
- Scatter Plot
- Box Plot
- Correlation Heatmap

### Correlation Analysis
- Pearson correlation matrix generated
- Spearman correlation matrix generated
- Pearson and Spearman correlations showed very small differences.
- Pearson correlation will be used for feature selection in Part 2.

### Grouped Aggregation
Grouped analysis was performed using Gender and Age.
- Highest Mean Group: Gender = 1
- Highest Standard Deviation Group: Gender = 0
- Mean Ratio = 1.0069

## Project Structure

```
NeuroSage/
│
├── data/
├── notebooks/
├── outputs/
├── figures/
├── docs/
├── app/
├── README.md
├── requirements.txt
└── .gitignore
```

## Next Step

Part 2 will focus on:
- Feature Engineering
- Data Preprocessing
- Train/Test Split
- Feature Scaling
- Machine Learning Model Development