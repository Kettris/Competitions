import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import os
import warnings
warnings.filterwarnings('ignore')


folder_path = r'C:\Users\nigil\OneDrive\Документы\прога\суббота\проект'
os.chdir(folder_path)

print("Предсказание популярности игр ")

# train (с категориями) и test (без них)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"\nразмер train: {train.shape}")
print(f"размер test: {test.shape}")

# целевая переменная (категория популярности)
target_col = 'popularity_category'
y = train[target_col].copy()
X_train = train.drop(columns=[target_col])
X_test = test.copy()

print(f"целевая переменная: {target_col} (категория популярности 0-5)")
print(f"распределение в обучении:")
for cat in range(6):
    print(f"   категория {cat}: {(y == cat).sum()} игр")


# инженерия признаков

print("\nсоздание новых признаков")

def add_new_features(df, is_train=True):
    df = df.copy()
    
    # цена (логарифм)
    if 'Price' in df.columns:
        median_price = df['Price'].median() if is_train else 0
        df['price_filled'] = df['Price'].fillna(median_price)
        df['log_price'] = np.log1p(df['price_filled'])
    else:
        df['log_price'] = 0
    
    # достижения
    if 'Achievements' in df.columns:
        df['achievements'] = df['Achievements'].fillna(0)
        df['log_achievements'] = np.log1p(df['achievements'])
    else:
        df['log_achievements'] = 0
    
    # рекомендации
    if 'Recommendations' in df.columns:
        df['recommendations'] = df['Recommendations'].fillna(0)
        df['log_recommendations'] = np.log1p(df['recommendations'])
    else:
        df['log_recommendations'] = 0
    
    # отзывы (доля положительных)
    if 'Positive' in df.columns and 'Negative' in df.columns:
        df['positive'] = df['Positive'].fillna(0)
        df['negative'] = df['Negative'].fillna(0)
        df['total_reviews'] = df['positive'] + df['negative']
        df['positive_ratio'] = df['positive'] / (df['total_reviews'] + 1)
    else:
        df['positive_ratio'] = 0.5
    
    # среднее время игры
    if 'Average playtime forever' in df.columns:
        df['playtime'] = df['Average playtime forever'].fillna(0)
        df['log_playtime'] = np.log1p(df['playtime'])
    else:
        df['log_playtime'] = 0
    
    # количество разработчиков и издателей
    if 'Developers' in df.columns:
        dev_str = df['Developers'].fillna('').astype(str)
        df['num_developers'] = dev_str.str.count(',') + 1
        df.loc[dev_str == '', 'num_developers'] = 0
    else:
        df['num_developers'] = 0
    
    if 'Publishers' in df.columns:
        pub_str = df['Publishers'].fillna('').astype(str)
        df['num_publishers'] = pub_str.str.count(',') + 1
        df.loc[pub_str == '', 'num_publishers'] = 0
    else:
        df['num_publishers'] = 0
    
    # поддержка платформ
    for platform in ['Windows', 'Mac', 'Linux']:
        if platform in df.columns:
            df[f'plat_{platform.lower()}'] = df[platform].fillna(0).astype(int)
        else:
            df[f'plat_{platform.lower()}'] = 0
    
    # жанры и теги (текст)
    if 'Genres' in df.columns:
        df['genres_str'] = df['Genres'].fillna('').astype(str).str.lower()
    else:
        df['genres_str'] = ''
    
    if 'Tags' in df.columns:
        df['tags_str'] = df['Tags'].fillna('').astype(str).str.lower()
    else:
        df['tags_str'] = ''
    
    # длина описания
    if 'About the game' in df.columns:
        df['desc_len'] = df['About the game'].fillna('').astype(str).str.len()
        df['log_desc'] = np.log1p(df['desc_len'])
    else:
        df['log_desc'] = 0
    
    # количество языков
    if 'Supported languages' in df.columns:
        lang_str = df['Supported languages'].fillna('').astype(str)
        df['num_languages'] = lang_str.str.count(',') + 1
        df.loc[lang_str == '', 'num_languages'] = 0
    else:
        df['num_languages'] = 0
    
    # возрастной рейтинг
    if 'Required age' in df.columns:
        df['required_age'] = df['Required age'].fillna(0)
        df['is_adult'] = (df['required_age'] >= 18).astype(int)
    else:
        df['is_adult'] = 0
    
    # наличие скидок
    if 'DiscountDLC count' in df.columns:
        df['has_discount'] = (df['DiscountDLC count'] > 0).astype(int)
    else:
        df['has_discount'] = 0
    
    # есть ли видео
    if 'Movies' in df.columns:
        df['has_movies'] = df['Movies'].notna().astype(int)
    else:
        df['has_movies'] = 0
    
    # есть ли скриншоты
    if 'Screenshots' in df.columns:
        df['has_screenshots'] = df['Screenshots'].notna().astype(int)
    else:
        df['has_screenshots'] = 0
    
    # количество категорий
    if 'Categories' in df.columns:
        cat_str = df['Categories'].fillna('').astype(str)
        df['num_categories'] = cat_str.str.count(',') + 1
        df.loc[cat_str == '', 'num_categories'] = 0
    else:
        df['num_categories'] = 0
    
    return df

X_train = add_new_features(X_train, is_train=True)
X_test = add_new_features(X_test, is_train=False)

# список признаков
feature_cols = [
    'log_price', 'log_achievements', 'log_recommendations',
    'positive_ratio', 'log_playtime', 'num_developers', 'num_publishers',
    'plat_windows', 'plat_mac', 'plat_linux', 'num_languages', 'log_desc',
    'is_adult', 'has_discount', 'has_movies', 'has_screenshots', 'num_categories',
    'genres_str', 'tags_str'
]

print(f"использование {len(feature_cols)} признаков")



# подготовка данных


numeric_cols = [c for c in feature_cols if c not in ['genres_str', 'tags_str']]
text_cols = ['genres_str', 'tags_str']

print(f"числовые признаке: {len(numeric_cols)}")
print(f"текстовые признаки: {len(text_cols)}")

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

text_processors = []
for col in text_cols:
    text_processors.append((
        col,
        CountVectorizer(max_features=100, stop_words='english', lowercase=True, min_df=2),
        col
    ))

preprocessor = ColumnTransformer(
    [('num', num_pipeline, numeric_cols)] + text_processors
)



# выбор лучшей модели

print("\nПробы разных моделей")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
}

best_model = None
best_f1 = 0
best_name = ""

for name, model in models.items():
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])
    
    pipeline.fit(X_train, y)
    y_pred = pipeline.predict(X_train)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    
    print(f"   {name}: accuracy = {acc:.4f}, f1 = {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_name = name

print(f"\nлучшая модель: {best_name} (f1 = {best_f1:.4f})")



# предсказание

print("\nПрредсказывание категории популярности для игр")

y_pred = best_model.predict(X_test)

# статистика предсказаний
print(f"\nКатегории предсказывания:")
for cat in range(6):
    count = (y_pred == cat).sum()
    print(f"   категория {cat}: {count} игр ({count/len(y_pred)*100:.1f}%)")



results = pd.DataFrame({
    'index': X_test.index,
    'popularity_category': y_pred
})

results.to_csv('results.csv', index=False)

print(f"\nрезультат сохранён в results.csv")
print(f"   всего предсказаний: {len(results)}")



# важность признаков (для random forest)

if best_name == 'Random Forest':
    print("\nважность признаков (топ-10):")
    try:
        rf = best_model.named_steps['clf']
        importances = rf.feature_importances_[:len(numeric_cols)]
        importance_list = list(zip(numeric_cols, importances))
        importance_list.sort(key=lambda x: x[1], reverse=True)
        
        for name, imp in importance_list[:10]:
            print(f"   {name:25s}: {imp:.4f}")
    except:
        print("   (не получилось посчитать важность)")

print("категории популярности:")
print("   0: 0 владельцев")
print("   1: 1 - 999 владельцев")
print("   2: 1,000 - 9,999 владельцев")
print("   3: 10,000 - 99,999 владельцев")
print("   4: 100,000 - 999,999 владельцев")
print("   5: 1,000,000+ владельцев")