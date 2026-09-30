import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Tải tập dữ liệu phân lớp nhị phân
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Chia tập dữ liệu Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Mô hình Cây quyết định ID3 (Criterion = 'entropy')
id3_model = DecisionTreeClassifier(
    criterion='entropy',  # Tiêu chí ID3 (Entropy & Information Gain)
    max_depth=5,          # Giới hạn độ sâu để tránh overfitting
    random_state=42
)

# 4. Huấn luyện mô hình
id3_model.fit(X_train, y_train)

# 5. Dự báo trên tập kiểm thử
y_pred = id3_model.predict(X_test)

# 6. Tính toán các độ đo đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 45)
print(" KẾT QUẢ PHÂN LỚP NHỊ PHÂN VỚI CÂY QUYẾT ĐỊNH ID3")
print("=" * 45)
print(f"1. Accuracy  (Độ chính xác toàn cục) : {accuracy * 100:.2f}%")
print(f"2. Precision (Độ chính xác)          : {precision * 100:.2f}%")
print(f"3. Recall    (Độ nhạy)                : {recall * 100:.2f}%")
print(f"4. F1-Score  (Trung bình điều hòa)   : {f1 * 100:.2f}%")
print("=" * 45)
print("\nBáo cáo chi tiết (Classification Report):")
print(classification_report(y_test, y_pred, target_names=data.target_names))