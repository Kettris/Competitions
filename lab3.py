import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import scipy.sparse as sp

# загрузка данных
train = pd.read_csv(r'C:\Users\nigil\OneDrive\Документы\прога\суббота\3\train.csv')
test = pd.read_csv(r'C:\Users\nigil\OneDrive\Документы\прога\суббота\3\test_x.csv')

# предобработка, целевая переменная
y_train = train['salary_mean_net'].values

# признаки "инвалидность/дети" перевод в числа
for col in ['accept_handicapped', 'accept_kids']:
    train[col] = train[col].astype(str).map({'True': 1, 'False': 0}).fillna(0)
    test[col] = test[col].astype(str).map({'True': 1, 'False': 0}).fillna(0)

num_features = ['accept_handicapped', 'accept_kids']

# name и professional_roles дают больше всего инфы о зарплате, а длинные описания дают много шума (их меньше)

text_features_to_use = {
    'name': 150,
    'professional_roles_name': 100,
    'key_skills_name': 50,
    'lemmaized_wo_stopwords_raw_description': 50, 
}

text_parts = []
for col, max_features in text_features_to_use.items():
    if col in train.columns:
        # запол пропуски и объединене поля для train и test
        train_text = train[col].fillna('').astype(str)
        test_text = test[col].fillna('').astype(str)
        
        # векторизуем каждое поле отдельно
        tfidf = TfidfVectorizer(max_features=max_features, 
                                ngram_range=(1, 2), 
                                min_df=2,
                                stop_words='english') # англ стоп-слова
        train_vec = tfidf.fit_transform(train_text)
        test_vec = tfidf.transform(test_text)
        text_parts.append((train_vec, test_vec))
        print(f"Обработано текстовое поле: {col}")

# обработка остав текстовых поле (с меньшим числом признаков)
other_text_cols = ['raw_description', 'lemmaized_wo_stopwords_raw_branded_description']
for col in other_text_cols:
    if col in train.columns:
        train_text = train[col].fillna('').astype(str)
        test_text = test[col].fillna('').astype(str)
        tfidf = TfidfVectorizer(max_features=30, ngram_range=(1, 2), min_df=2)
        train_vec = tfidf.fit_transform(train_text)
        test_vec = tfidf.transform(test_text)
        text_parts.append((train_vec, test_vec))
        print(f"Обработано дополнительное текстовое поле: {col}")

# доб категор признаков
categorical_features = [
    'experience_name',  # опыт: (для зарплаты)
    'schedule_name',    # график: полный день, удаленка итд
    'unified_address_city', # город: зарплаты в Москве и в регионе отличаются
    'unified_address_state',# область/край
    'employment_name',      # тип занятости (полная, частичная)
]

# заполнение пропусков
for col in categorical_features:
    if col in train.columns:
        train[col] = train[col].fillna('unknown').astype(str)
        test[col] = test[col].fillna('unknown').astype(str)

# код категор признаки
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
train_cat = ohe.fit_transform(train[categorical_features])
test_cat = ohe.transform(test[categorical_features])

branded_feature = []
if 'is_branded_description' in train.columns:
    # заполн пропуски
    train_branded = train['is_branded_description'].fillna('не заполнено')
    test_branded = test['is_branded_description'].fillna('не заполнено')
    # в числа "заполнено" -> 1, остальное -> 0
    train_branded = (train_branded == 'заполнено').astype(int)
    test_branded = (test_branded == 'заполнено').astype(int)
    branded_feature = [train_branded.values.reshape(-1, 1), test_branded.values.reshape(-1, 1)]

# обьединяем все признаки в одну большую матрицу
X_train_parts = []
X_test_parts = []

# доб обработанные тексты
for train_vec, test_vec in text_parts:
    X_train_parts.append(train_vec)
    X_test_parts.append(test_vec)

# доб категор признаки
X_train_parts.append(train_cat)
X_test_parts.append(test_cat)

# доб числовые признаки
X_train_parts.append(train[num_features].values)
X_test_parts.append(test[num_features].values)

# доб признак is_branded_description
if branded_feature:
    X_train_parts.append(branded_feature[0])
    X_test_parts.append(branded_feature[1])

# склеивание все части в одну матрицу
import scipy.sparse as sp
X_train = sp.hstack(X_train_parts).tocsr()
X_test = sp.hstack(X_test_parts).tocsr()

# преобразование целевой переменной (логарифмирование)
y_train_log = np.log1p(y_train)

# настройка модели RandomForest
model = RandomForestRegressor(n_estimators=200, 
                              max_depth=25, 
                              min_samples_split=5, 
                              min_samples_leaf=3, 
                              random_state=42, 
                              n_jobs=-1)
model.fit(X_train, y_train_log)

# предсказание и обратное преобразование
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)

# постобработка предсказаний (Clipping)
preds = np.clip(preds, 15000, 250000)

output = pd.DataFrame({'id': test['id'], 'salary_mean_net': preds})
output.to_csv(r'C:\Users\nigil\OneDrive\Документы\прога\суббота\3\results.csv', index=False)

