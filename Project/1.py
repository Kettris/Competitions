import pandas as pd
import numpy as np
import os
import re

folder_path = r'C:\Users\nigil\OneDrive\Документы\прога\суббота\проект'

df = pd.read_csv(os.path.join(folder_path, 'games.csv'))

print("Оригинальные названия колонок:")
print(df.columns.tolist())
print("="*50)

# функция для преобразования Estimated owners в число
def parse_estimated_owners(value):
    if pd.isna(value):
        return np.nan
    value_str = str(value)
    match = re.search(r'(\d+)\s*-\s*(\d+)', value_str)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) / 2
    match_single = re.search(r'(\d+)', value_str)
    if match_single:
        return int(match_single.group(1))
    return np.nan

# преобразование в числа
df['Estimated_owners_numeric'] = df['Estimated owners'].apply(parse_estimated_owners)

# только строки с числами
df_clean = df[df['Estimated_owners_numeric'].notna()].copy()

print(f"\nИсходный размер: {len(df)}")
print(f"Размер после очистки: {len(df_clean)}")

# создание категории популярности
def popularity_category(owners):
    if owners == 0:
        return 0
    elif owners < 1000:
        return 1
    elif owners < 10000:
        return 2
    elif owners < 100000:
        return 3
    elif owners < 1000000:
        return 4
    else:
        return 5

df_clean['popularity_category'] = df_clean['Estimated_owners_numeric'].apply(popularity_category)

print(f"\nРаспределение категорий популярности:")
for cat in range(6):
    count = (df_clean['popularity_category'] == cat).sum()
    pct = count / len(df_clean) * 100
    if cat == 0:
        name = "0 владельцев"
    elif cat == 1:
        name = "1-999"
    elif cat == 2:
        name = "1k-9.9k"
    elif cat == 3:
        name = "10k-99k"
    elif cat == 4:
        name = "100k-999k"
    else:
        name = "1M+"
    print(f"   Категория {cat} ({name:12}): {count:>6} игр ({pct:5.1f}%)")

# перемешиваем
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

# деление 20% train, 80% test
split_idx = int(0.2 * len(df_clean))
train_df = df_clean.iloc[:split_idx].copy()
test_df = df_clean.iloc[split_idx:].copy()

# в train_df оставляем категорию как целевую переменную
train_df = train_df.drop(columns=['Estimated owners', 'Estimated_owners_numeric'])
train_df.rename(columns={'popularity_category': 'popularity_category'}, inplace=True)

# в test_df удаляем всё что обозн с целевой переменной
test_df_no_target = test_df.drop(columns=['Estimated owners', 'Estimated_owners_numeric', 'popularity_category'])

train_df.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
test_df_no_target.to_csv(os.path.join(folder_path, 'test.csv'), index=False)

print(f"Train size: {len(train_df)} (20% данных)")
print(f"Test size: {len(test_df_no_target)} (80% данных)")
print(f"\nЦелевая переменная: popularity_category (0-5, где 5 = самый популярный)")