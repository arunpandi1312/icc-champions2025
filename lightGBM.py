import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb


data_path = "/content/Feature_Engineered_Data_ICC.csv"
df = pd.read_csv(data_path)


df.fillna(df.median(), inplace=True)


if df['Outcome.1'].dtype != 'int64' and df['Outcome.1'].dtype != 'object':
    df['Outcome.1'] = LabelEncoder().fit_transform(df['Outcome.1'])


X = df.drop(columns=['Outcome.1'])
y = df['Outcome.1']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
