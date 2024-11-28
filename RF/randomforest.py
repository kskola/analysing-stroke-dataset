import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


#collect and clean data
df = pd.read_csv('C:/analysing stroke dataset/stroke_data.csv')
df = df.dropna()
#check data
print(df)

#create the model
model = RandomForestClassifier()

# Select features (X) and target variable (y)
X = df[['sex','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']]
y = df['stroke']
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(class_weight='balanced',n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'The Model Accuracy = {accuracy}')

#testthe model
prediction = clf.predict(X)
print('the model score =', clf.score(X,y))

# Function to calculate sensitivity and specificity
def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    return sensitivity, specificity

# Random Forest metrics
rf_sensitivity, rf_specificity = calculate_sensitivity_specificity(y_test, y_pred)
print("Random Forest Sensitivity: ", rf_sensitivity)
print("Random Forest Specificity: ", rf_specificity)

# Calculate the mean age of the patients
mean_age = df['age'].mean()
print(f'Mean age of patients: {mean_age}')

# Calculate the standard deviation of the age
std_age = df['age'].std()
print(f'Standard deviation of age: {std_age}')

# Calculate the percentage of patients with hypertension
percent_hypertension = (df['hypertension'].sum() / len(df)) * 100
print(f'Percentage of patients with hypertension: {percent_hypertension:.2f}%')

# Calculate the percentage of patients with heart disease
percent_heart_disease = (df['heart_disease'].sum() / len(df)) * 100
print(f'Percentage of patients with heart disease: {percent_heart_disease:.2f}%')

# Calculate the percentage of smokers
percent_smokers = (df['smoking_status'] == 1).sum() / len(df) * 100
print(f'Percentage of smokers: {percent_smokers:.2f}%')

# Calculate the average of the average glucose level
mean_avg_glucose_level = df['avg_glucose_level'].mean()
print(f'Average of average glucose level: {mean_avg_glucose_level}')

# Calculate the standard deviation of the average glucose level
std_avg_glucose_level = df['avg_glucose_level'].std()
print(f'Standard deviation of average glucose level: {std_avg_glucose_level}')

gender_counts = df['sex'].value_counts(normalize=True) * 100
male_percentage = gender_counts.get(1, 0)
female_percentage = gender_counts.get(0, 0)

print(f"Percentage of Male Patients: {male_percentage}%")
print(f"Percentage of Female Patients: {female_percentage}%")



#show the proportion of stroke vs. no stroke in the dataset.
sns.countplot(x='stroke', data=df)
plt.title('Distribution of Stroke Incidence')
plt.show()

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

 #Feature importance
feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importance:")
print(feature_importances)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # To have the most important feature at the top
plt.show()

### 1. Exploratory Data Analysis (EDA)

# Distribution of Target Variable (Stroke Incidence)**
#Bar plot or pie chart
#To show the proportion of stroke vs. no stroke in the dataset.

import seaborn as sns

sns.countplot(x='stroke', data=df)
plt.title('Distribution of Stroke Incidence')
plt.show()

# Feature Distributions**
# Histograms or KDE plots
#To visualize the distribution of numerical features.

df.hist(bins=20, figsize=(15, 10), edgecolor='black')
# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.show()

# Correlation Matrix
#Heatmap
#To show the correlation between different features.

import seaborn as sns

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

### 2. Model Evaluation

#Confusion Matrix**
#Heatmap
#To visualize the performance of the classification model.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

#ROC Curve
#To visualize the trade-off between sensitivity and specificity.


# Get predicted probabilities for the positive class (stroke)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Random Forest ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guessing')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


 #Precision-Recall Curve
#Precision-Recall Curve

from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


#prediction for a new patient
#new_patient = [[1, 85.0, 0, 0, 1, 4, 1, 186.21, 28.0, 1]]  # Example values for age, bmi, heart_disease, avg_glucose_level, hypertension, smoking_status
#prediction = clf.predict(new_patient)
#if prediction[0] == 1:
   # print('Prediction: The patient is likely to have a stroke.')
#else:
   # print('Prediction:The patient is not likely to have a stroke.')

