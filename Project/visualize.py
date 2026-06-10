import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import re


folder_path = r'C:\Users\nigil\OneDrive\Документы\прога\суббота\проект'
os.chdir(folder_path)

# стиль графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("="*60)
print("визуализация проекта: предсказание популярности игр")
print("="*60)

original = pd.read_csv('games.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
results = pd.read_csv('results.csv')

print(f"\nзагружено:")
print(f"   исходный датасет: {len(original)} игр")
print(f"   train (обучение):  {len(train)} игр (20%)")
print(f"   test (предсказание): {len(test)} игр (80%)")
print(f"   результаты:         {len(results)} предсказаний")

# функция для преобразования estimated owners

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

# доб категории в оригинал
original['owners_numeric'] = original['Estimated owners'].apply(parse_estimated_owners)
original['category'] = original['owners_numeric'].apply(popularity_category)
original_clean = original[original['owners_numeric'].notna()].copy()

# восстаy разбиение (как в 1.py)
np.random.seed(42)
indices = np.random.permutation(len(original_clean))
split_idx = int(0.2 * len(original_clean))
test_indices = indices[split_idx:]

y_true = original_clean.iloc[test_indices]['category'].values
y_pred = results['popularity_category'].values

# график 1: распределение игр по категориям (train vs test)

print("\n1. строю график распределения категорий")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# распределение в train
train_cats = train['popularity_category'].value_counts().sort_index()
axes[0].bar(train_cats.index, train_cats.values, color='steelblue', edgecolor='black')
axes[0].set_xlabel('категория популярности')
axes[0].set_ylabel('количество игр')
axes[0].set_title('распределение в train (20% данных)\nмодель учится на этих играх')
axes[0].set_xticks(range(6))
axes[0].grid(axis='y', alpha=0.3)

# подписи на столбцах
for i, (cat, count) in enumerate(train_cats.items()):
    axes[0].text(cat, count + 5, str(count), ha='center', fontsize=10)

# распределение в test
test_real_cats = pd.Series(y_true).value_counts().sort_index()
axes[1].bar(test_real_cats.index, test_real_cats.values, color='coral', edgecolor='black')
axes[1].set_xlabel('категория популярности')
axes[1].set_ylabel('количество игр')
axes[1].set_title('распределение в test (80% данных)\nмодель проверяется на этих играх')
axes[1].set_xticks(range(6))
axes[1].grid(axis='y', alpha=0.3)

for i, (cat, count) in enumerate(test_real_cats.items()):
    axes[1].text(cat, count + 5, str(count), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plot1_distribution_train_test.png', dpi=150, bbox_inches='tight')
plt.show()
print("   сохранено: plot1_distribution_train_test.png")

# график 2: реальные vs предсказанные категории

print("\n2. строю график сравнения реальных и предсказанных категорий")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# реальные категории
real_counts = pd.Series(y_true).value_counts().sort_index()
axes[0].bar(real_counts.index, real_counts.values, color='forestgreen', alpha=0.7, edgecolor='black', label='реальные')
axes[0].set_xlabel('категория популярности')
axes[0].set_ylabel('количество игр')
axes[0].set_title('реальное распределение категорий в test')
axes[0].set_xticks(range(6))
axes[0].grid(axis='y', alpha=0.3)

# предсказанные категории
pred_counts = pd.Series(y_pred).value_counts().sort_index()
axes[1].bar(pred_counts.index, pred_counts.values, color='darkorange', alpha=0.7, edgecolor='black', label='предсказанные')
axes[1].set_xlabel('категория популярности')
axes[1].set_ylabel('количество игр')
axes[1].set_title('предсказанное распределение категорий в test')
axes[1].set_xticks(range(6))
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot2_real_vs_pred_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("   сохранено: plot2_real_vs_pred_distribution.png")

# график 3: матрица ошибок (confusion matrix)

print("\n3. строю матрицу ошибок")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=[f'pred {i}' for i in range(6)],
            yticklabels=[f'real {i}' for i in range(6)])
ax.set_xlabel('предсказанная категория')
ax.set_ylabel('реальная категория')
ax.set_title('матрица ошибок (confusion matrix)\n'
             'диагональ — правильные предсказания, остальное — ошибки')

plt.tight_layout()
plt.savefig('plot3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("   сохранено: plot3_confusion_matrix.png")

# график 4: точность по каждой категории

print("\n4. строю график точности по категориям")

category_names = ['0\n(0)', '1\n(1-999)', '2\n(1k-9.9k)', '3\n(10k-99k)', '4\n(100k-999k)', '5\n(1M+)']
precision_by_cat = []

for cat in range(6):
    real_cat = (y_true == cat)
    pred_cat = (y_pred == cat)
    correct = (real_cat & pred_cat).sum()
    pred_count = pred_cat.sum()
    precision = correct / pred_count if pred_count > 0 else 0
    precision_by_cat.append(precision)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(6), precision_by_cat, color='teal', edgecolor='black')
ax.set_xlabel('категория популярности')
ax.set_ylabel('точность (precision)')
ax.set_title('точность предсказаний по каждой категории\n'
             'сколько предсказанных игр категории X действительно относятся к ней')
ax.set_xticks(range(6))
ax.set_xticklabels(category_names)
ax.set_ylim(0, 1)
ax.axhline(y=0.1667, color='red', linestyle='--', label='случайное угадывание (16.7%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

for i, (bar, prec) in enumerate(zip(bars, precision_by_cat)):
    ax.text(i, prec + 0.02, f'{prec:.1%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plot4_precision_by_category.png', dpi=150, bbox_inches='tight')
plt.show()
print("   сохранено: plot4_precision_by_category.png")
