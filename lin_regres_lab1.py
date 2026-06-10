import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

folder_path = r'C:\Users\nigil\OneDrive\Документы\прога\суббота'

train = pd.read_csv(os.path.join(folder_path, 'prices_train.csv'), index_col=0)
test = pd.read_csv(os.path.join(folder_path, 'prices_test.csv'), index_col=0)

X = train.drop('Y house price of unit area', axis=1)
y = train['Y house price of unit area']

X['X2_log'] = np.log(X['X2 house age'] + 1)
X['X3_log'] = np.log(X['X3 distance to the nearest MRT station'] + 1)
X['X4_squared'] = X['X4 number of convenience stores'] ** 2
X['age_times_stores'] = X['X2 house age'] * X['X4 number of convenience stores']

test['X2_log'] = np.log(test['X2 house age'] + 1)
test['X3_log'] = np.log(test['X3 distance to the nearest MRT station'] + 1)
test['X4_squared'] = test['X4 number of convenience stores'] ** 2
test['age_times_stores'] = test['X2 house age'] * test['X4 number of convenience stores']

features = ['X1 transaction date', 'X2_log', 'X3_log', 'X4 number of convenience stores',
            'X4_squared', 'X5 latitude', 'X6 longitude', 'age_times_stores']

X = X[features]
test_df = test[features].copy()

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
test = imputer.transform(test_df)

scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)

model = LinearRegression()
model.fit(X, y)

pred = model.predict(test)

result = pd.DataFrame({
    'index': test_df.index,
    'Y house price of unit area': pred
})

output_path = os.path.join(folder_path, 'result.csv')
result.to_csv(output_path, index=False)

print(result.head(85))

