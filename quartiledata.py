# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = r'C:\Users\craig\OneDrive\Documents\cs\ml_proj\finalfinal.csv'
data = pd.read_csv(file_path)

# Step 1: Calculate Quartiles for the 'CGPA' column
q1 = data['CGPA'].quantile(0.25)  # 25th percentile
q2 = data['CGPA'].quantile(0.50)  # 50th percentile (Median)
q3 = data['CGPA'].quantile(0.75)  # 75th percentile

print(f"Quartile 1 (25%): {q1}")
print(f"Quartile 2 (Median, 50%): {q2}")
print(f"Quartile 3 (75%): {q3}")

# Step 2: Classify CGPA into Quartile-Based Groups
data['CGPA_Quartile_Class'] = pd.cut(
    data['CGPA'], 
    bins=[-np.inf, q1, q3, np.inf], 
    labels=['Bottom 25%', 'Interquartile Range', 'Top 25%']
)

# Display the first few rows to check the classification
print("\nSample of classified CGPA quartile classes:")
print(data[['CGPA', 'CGPA_Quartile_Class']].head())

# Step 3: Prepare Data for Machine Learning Model
# Drop irrelevant columns that are identifiers, non-numeric, or contain target values
data_cleaned = data.copy()
data_cleaned['CGPA_Quartile_Class'] = data_cleaned['CGPA_Quartile_Class'].astype(str)

# Drop columns that are not features
X = data_cleaned.drop(columns=['EmployeeName', 'UserId', 'CGPA',  'CGPA_Quartile_Class'])
y = data_cleaned['CGPA_Quartile_Class']



# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Classification Model (Random Forest)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 7: Predict on the Test Set
y_pred = rf_classifier.predict(X_test)

# Step 8: Evaluate the Model's Performance
classification_results = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print the classification report and accuracy
print("\nClassification Report for Quartile-Based Classification:")
print(classification_results)
print(f"\nOverall Accuracy: {accuracy:.2f}")
