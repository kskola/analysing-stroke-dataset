Stroke Prediction Model
This repository contains a Python implementation of a machine learning model to predict the likelihood of stroke occurrence based on various health metrics. The model utilizes a Random Forest Classifier and performs exploratory data analysis (EDA) to understand the dataset better.


Data
The dataset used in this project is a CSV file containing various health-related features of patients, including:

sex: Gender of the patient
age: Age of the patient
hypertension: Hypertension status (1: Yes, 0: No)
heart_disease: Heart disease status (1: Yes, 0: No)
ever_married: Marital status (1: Yes, 0: No)
work_type: Type of work
Residence_type: Type of residence
avg_glucose_level: Average glucose level
bmi: Body Mass Index
smoking_status: Smoking status
stroke: Target variable (1: Stroke, 0: No Stroke)

Model
The model is built using the Random Forest Classifier from the scikit-learn library. The following steps are performed in the model:

Data Cleaning: Remove any rows with missing values.
Feature Selection: Select relevant features and the target variable.
Train-Test Split: Split the data into training and testing sets (80% train, 20% test).
Model Training: Train the Random Forest model on the training data.
Prediction: Make predictions on the test set.
Evaluation: Calculate accuracy, sensitivity, specificity, and generate a classification report.
Results
The model's performance metrics include:

Model Accuracy
Sensitivity
Specificity
Confusion Matrix
Classification Report
The model also provides insights into the distribution of various features within the dataset.

Visualizations
The following visualizations are generated to help understand the data and model performance:

Count plot of stroke incidence.
Histograms of numerical features.
Correlation matrix heatmap.
Confusion matrix heatmap.
ROC curve.
Precision-Recall curve.
Feature importance bar chart.
