import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns


col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
df = pd.read_csv('adult.data',header = None, names = col_names)

#Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()
print(df.head())

#1. Check Class Imbalance


#2. Create feature dataframe X with feature columns and dummy variables for categorical features
feature_cols = ['age','capital-gain', 'capital-loss', 'hours-per-week', 'sex','race', 'hours-per-week', 'education']


#3. Create a heatmap of X data to see feature correlation


#4. Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greater than 50k


#5a. Split data into a train and test set


#5b. Fit LR model with sklearn on train set, and predicting on the test set
log_reg = LogisticRegression(C=0.05, penalty='l1', solver='liblinear')


#6. Print model parameters (intercept and coefficients)
print('Model Parameters, Intercept:')

print('Model Parameters, Coeff:')


#7. Evaluate the predictions of the model on the test set. Print the confusion matrix and accuracy score.
print('Confusion Matrix on test set:')
print('Accuracy Score on test set:')

# 8.Create new DataFrame of the model coefficients and variable names; sort values based on coefficient

#9. barplot of the coefficients sorted in ascending order


#10. Plot the ROC curve and print the AUC value.
#y_pred_prob = log_reg.predict_proba(x_test)

print(df['income'].value_counts())

X = pd.get_dummies(df[feature_cols], drop_first=True)

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Create the output variable y
y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit the logistic regression model
log_reg.fit(x_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(x_test)

# Print the model intercept
print('Model Parameters, Intercept:', log_reg.intercept_)

# Print the model coefficients
print('Model Parameters, Coefficients:', log_reg.coef_)


from sklearn.metrics import confusion_matrix, accuracy_score

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix on test set:')
print(conf_matrix)

# Print the accuracy score
accuracy = log_reg.score(x_test, y_test)
print('Accuracy Score on test set:', accuracy)


# Pair variable names with coefficients
coefficients = zip(X.columns, log_reg.coef_[0])

# Create a DataFrame
coeff_df = pd.DataFrame(coefficients, columns=['Variable', 'Coefficient'])

# Sort the DataFrame by coefficient values
coeff_df = coeff_df[coeff_df['Coefficient'] != 0].sort_values(by='Coefficient')

# Print the sorted DataFrame
print(coeff_df)


# Create a barplot of the coefficients
plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Variable', data=coeff_df, palette='viridis')
plt.title('Coefficients of Logistic Regression Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Variable')
plt.show()


# Get predicted probabilities for the positive class
y_pred_prob = log_reg.predict_proba(x_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate AUC
auc_value = roc_auc_score(y_test, y_pred_prob)
print('AUC:', auc_value)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_value:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
