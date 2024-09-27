import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train_data = pd.read_csv(r'C:\Users\syama\Downloads\fraudTrain.csv').sample(frac=0.2, random_state=42) 
test_data = pd.read_csv(r'C:\Users\syama\Downloads\fraudTest.csv')
print(train_data.head())
numeric_columns = train_data.select_dtypes(include=[np.number]).columns 
plt.figure(figsize=(12,8))
sns.heatmap(train_data[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Columns Only)')
plt.show()
X = train_data.drop(columns=['is_fraud', 'trans_num', 'dob', 'merchant', 'category', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 'trans_date_trans_time'])
y = train_data['is_fraud']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_reg = LogisticRegression(max_iter=1000)  
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
tree_clf = DecisionTreeClassifier(random_state=42, max_depth=10)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
forest_clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42) 
forest_clf.fit(X_train, y_train)
y_pred_forest = forest_clf.predict(X_test)

# Model Evaluation: Logistic Regression
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Model Evaluation: Decision Tree
print("Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))

# Model Evaluation: Random Forest
print("Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_forest)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_forest))
print("Classification Report:\n", classification_report(y_test, y_pred_forest))

# Compare model performance visually
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_score(y_test, y_pred_log_reg), accuracy_score(y_test, y_pred_tree), accuracy_score(y_test, y_pred_forest)]

plt.figure(figsize=(8,6))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.show()
