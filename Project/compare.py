import pandas as pd
import numpy as np
import os
import re

folder_path = r'C:\Users\nigil\OneDrive\Документы\прога\суббота\проект'
os.chdir(folder_path)

print("\n1. Загрузка исходных данных")
original = pd.read_csv('games.csv')
print(f"   исходный датасет: {original.shape[0]} строк, {original.shape[1]} колонок")

print("\n2. Загрузка предсказания модели")
results = pd.read_csv('results.csv')
print(f"   предсказания: {len(results)} строк")



# функция для преобразования estimated owners в число

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



# функция для определения категории популярности

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



# восстанавление строк какие мы предсказывали

print("\n3. Востановление, какие строки предсказывала модель")

# преобразов ориг датасет
original['estimated_owners_numeric'] = original['Estimated owners'].apply(parse_estimated_owners)
original['popularity_category'] = original['estimated_owners_numeric'].apply(popularity_category)

# убираем строки без данных (не в обучении)
original_clean = original[original['estimated_owners_numeric'].notna()].copy()

# перемешиваем и делим (как в 1.py)и повтор ту же логику (узанать какие строи попали в test)
np.random.seed(42)
indices = np.random.permutation(len(original_clean))
split_idx = int(0.2 * len(original_clean))

# индексы test-строк в original_clean
test_indices_in_clean = indices[split_idx:]

# реальные значения для test
y_true = original_clean.iloc[test_indices_in_clean]['popularity_category'].values
y_pred = results['popularity_category'].values

print(f"   всего сравнённых игр: {len(y_true)}")



# считаем метрики

print("\n4. Подсчет метрики качества")

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n" + "="*60)
print("Результаты сравнения")
print(f"\nколичество сравнённых игр: {len(y_true)}")
print(f"\nосновные метрики:")
print(f"   accuracy (точность): {acc:.4f} ({acc*100:.1f}%)")
print(f"   f1-score (взвешенный): {f1:.4f}")

# проверка для оценки качества 
print(f"\nоценка модели:")
if acc > 0.6:
    print("   отлично, модель хорошо определяет популярность")
elif acc > 0.4:
    print("   неплохо, модель улавливает основные закономерности")
elif acc > 0.3:
    print("   средненько, можно добавить ещё признаков")
else:
    print("   плохо, модель почти не угадывает")



# матрица ошибок (confusion matrix)

print("\n5. Построение матрицы ошибок")

print("\nматрица ошибок (confusion matrix):")
print("   (строки — реальные, столбцы — предсказанные)")
print("   " + "-"*50)

cm = confusion_matrix(y_true, y_pred)
print("        ", end="")
for cat in range(6):
    print(f"  pred{cat}", end="")
print()

for i in range(6):
    print(f"  real{i}: ", end="")
    for j in range(6):
        print(f"{cm[i,j]:6d}", end="")
    print()



# анализ

print(f"\n6. Анализ каждой категории")

print(f"\nдетальный анализ по категориям:")
print(f"   категория | реальных | предсказано | точность")
print(f"   " + "-"*50)

for cat in range(6):
    real_count = (y_true == cat).sum()
    pred_count = (y_pred == cat).sum()
    correct = ((y_true == cat) & (y_pred == cat)).sum()
    precision = correct / pred_count if pred_count > 0 else 0
    
    if cat == 0:
        name = "0 владельцев       "
    elif cat == 1:
        name = "1-999             "
    elif cat == 2:
        name = "1k-9.9k           "
    elif cat == 3:
        name = "10k-99k           "
    elif cat == 4:
        name = "100k-999k         "
    else:
        name = "1m+               "
    
    print(f"   {cat} ({name}) | {real_count:7} | {pred_count:10} | {precision*100:5.1f}%")



# сравнивнение распределения

print(f"\n7. Сравнение распределения")

print(f"\nраспределение реальных категорий:")
for cat in range(6):
    count = (y_true == cat).sum()
    print(f"   категория {cat}: {count} игр ({count/len(y_true)*100:.1f}%)")

print(f"\nраспределение предсказанных категорий:")
for cat in range(6):
    count = (y_pred == cat).sum()
    print(f"   категория {cat}: {count} игр ({count/len(y_pred)*100:.1f}%)")



print("\n8. Итоги")

comparison_df = pd.DataFrame({
    'real_category': y_true,
    'predicted_category': y_pred,
    'is_correct': y_true == y_pred
})

comparison_df.to_csv('comparison_full.csv', index=False)

print("  полный отчёт в comparison_full.csv")


if acc > 0.45:
    print("\nмодель работает достаточно хорошо")
    print(f"   accuracy = {acc:.3f}")
else:
    print("\nмодель можно улучшить")
    print("   попробуй добавить больше признаков или потестировать другие модели")

print(f"\nсозданные файлы:")
print(f"   - comparison_full.csv   (полная таблица сравнения)")
print(f"   - results.csv   (предсказания)")

