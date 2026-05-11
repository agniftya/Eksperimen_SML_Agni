import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('restaurant_satisfaction_preprocessing/train_preprocessed.csv')
test_data = pd.read_csv('restaurant_satisfaction_preprocessing/test_preprocessed.csv')

X_train = train_data.drop('HighSatisfaction', axis=1)
y_train = train_data['HighSatisfaction']
X_test = test_data.drop('HighSatisfaction', axis=1)
y_test = test_data['HighSatisfaction']

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Akurasi Model Dasar: {accuracy_score(y_test, y_pred):.4f}")