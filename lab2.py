import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
import os

folder_path = r'C:\Users\nigil\OneDrive\Документы\прога\суббота\2'
os.chdir(folder_path)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Размер train:', train.shape)
print('Размер test:', test.shape)

y = train['Depression']
X = train.drop(['Depression', 'id'], axis=1)
test_ids = test['id']
X_test = test.drop('id', axis=1)

# созд нов признак (общий стресс)
X['Stress_Total'] = X['Academic Pressure'] + X['Work Pressure'] + X['Financial Stress']
X_test['Stress_Total'] = X_test['Academic Pressure'] + X_test['Work Pressure'] + X_test['Financial Stress']

# запполнение пропуски 0
X['Stress_Total'] = X['Stress_Total'].fillna(0)
X_test['Stress_Total'] = X_test['Stress_Total'].fillna(0)

# кодирование суицид мысли
X['Suicidal'] = X['Have you ever had suicidal thoughts ?']
X_test['Suicidal'] = X_test['Have you ever had suicidal thoughts ?']

X['Suicidal'] = X['Suicidal'].map({'Yes': 1, 'No': 0})
X_test['Suicidal'] = X_test['Suicidal'].map({'Yes': 1, 'No': 0})

# удаление нек-х признаки (эксперимент 1)
X = X.drop(['Study Satisfaction', 'Job Satisfaction'], axis=1)
X_test = X_test.drop(['Study Satisfaction', 'Job Satisfaction'], axis=1)

# кодирование категор признаки
cat_cols = X.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# заполение пропуски медианой
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(X)
X_test = imp.transform(X_test)

# масштабирование
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# выбор лучшей модели через валидацию
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(random_state=42)
}

best_model = None
best_f1 = 0

# раззделение для валидации
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nРезультаты валидации:")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    f1 = f1_score(y_val, pred_val)
    print(f"{name}: F1 = {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

print(f"\nЛучшая модель: {best_model.__class__.__name__} с F1 = {best_f1:.4f}")

# обучение лучш модель на всех данных
best_model.fit(X, y)
pred = best_model.predict(X_test)

sub = pd.DataFrame({'id': test_ids, 'Depression': pred})
sub.to_csv('results.csv', index=False)

print('\nresults.csv создан')
