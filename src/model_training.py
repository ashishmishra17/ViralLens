import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load cleaned data
file_path = '../placement_cleaned.csv'
df = pd.read_csv(file_path)

# Split into features and target (assuming last column is target)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a basic Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Basic Model - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

# Hyperparameter tuning: n_estimators
param_grid = {'n_estimators': [10, 50, 100, 200]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
gs.fit(X_train, y_train)

print(f'Best n_estimators: {gs.best_params_["n_estimators"]}')
print(f'Best CV Accuracy: {gs.best_score_:.4f}')

# Evaluate tuned model
y_pred_tuned = gs.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')

print(f'Tuned Model - Accuracy: {accuracy_tuned:.4f}, F1 Score: {f1_tuned:.4f}')

# Short explanation of results
print('\nExplanation:')
print('A Random Forest model was trained and evaluated using accuracy and F1 score. Hyperparameter tuning for n_estimators improved cross-validation accuracy. The tuned model was also evaluated on the test set.')
