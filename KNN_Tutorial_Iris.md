# 建立 KNN 教學 Markdown 文件
md_content = """# K-Nearest Neighbors (KNN) 教學文件 — 以 Python 物件導向與 scikit-learn 為例

## 📘 目錄
1. [簡介](#簡介)
2. [資料集介紹 — Iris](#資料集介紹--iris)
3. [KNN 理論概念](#knn-理論概念)
4. [以物件導向方式實作 KNN](#以物件導向方式實作-knn)
5. [以 scikit-learn 套件實作 KNN](#以-scikit-learn-套件實作-knn)
6. [模型評估與混淆矩陣](#模型評估與混淆矩陣)
7. [完整程式碼](#完整程式碼)
8. [參考資料](#參考資料)

---

## 📌 簡介
K-Nearest Neighbors（KNN）是一種簡單但非常實用的分類演算法。
基本概念為：
- 給定一筆未知資料
- 找出訓練集中「最接近」的 `k` 筆資料
- 使用這些鄰居的多數決結果來預測該資料的分類

---

## 🌸 資料集介紹 — Iris
我們將使用 **Iris dataset**（鳶尾花資料集），它是機器學習中最經典的資料集之一。

| 特徵名稱         | 說明         |
|------------------|--------------|
| sepal length     | 花萼長度     |
| sepal width      | 花萼寬度     |
| petal length     | 花瓣長度     |
| petal width      | 花瓣寬度     |
| label            | 花的種類（Setosa / Versicolor / Virginica） |

---

## 🧠 KNN 理論概念
1. 計算測試點與所有訓練點的距離（常用歐式距離）。
2. 選擇距離最近的 `k` 個點。
3. 多數決：根據鄰居的分類決定預測結果。

公式（歐式距離）：
\\[ d(x, y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2} \\]

---

## 🧰 以物件導向方式實作 KNN

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)
