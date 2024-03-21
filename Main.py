import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('./gender.csv')

data.columns = data.columns.str.strip()

categorical_cols = ['Occupation', 'Education Level', 'Marital Status', 'Favorite Color']
data = pd.get_dummies(data, columns=categorical_cols)

features = data.drop(['Gender', 'Unnamed: 9'], axis=1)
labels = data['Gender']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

joblib.dump(model, './gender_model.pkl')
joblib.dump(scaler, './scaler.pkl')
feature_names = list(features.columns)
joblib.dump(feature_names, './feature_names.pkl')

