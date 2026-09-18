import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Hàm lưu vết (serialize) một đối tượng Python (như model.pkl, preprocessor.pkl)
    vào đĩa cứng dưới dạng file binary (.pkl) để tái sử dụng sau này.
    """
    try:
        # Trích xuất đường dẫn thư mục chứa file
        dir_path = os.path.dirname(file_path)

        # Tự động tạo thư mục lưu trữ nếu chưa tồn tại (tránh lỗi FileNotFoundError)
        os.makedirs(dir_path, exist_ok=True)

        # Mở file ở chế độ Ghi Nhị Phân (Write Binary - "wb") và đóng đóng gói đối tượng
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Hàm duyệt qua từng mô hình, thực hiện GridSearch 3-Fold Cross Validation 
    để tìm bộ siêu tham số tốt nhất, sau đó tính và trả về báo cáo điểm R2.
    """
    try:
        report = {}

        # Lặp qua từng thuật toán trong Dictionary 'models'
        for i in range(len(list(models))):
            model = list(models.values())[i]
            # Lấy lưới tham số tương ứng của mô hình hiện tại
            para = param[list(models.keys())[i]]

            # Thiết lập GridSearch với Cross Validation k=3 để tìm tổ hợp tham số tốt nhất
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Cập nhật siêu tham số tối ưu vừa tìm được vào mô hình
            model.set_params(**gs.best_params_)
            
            # Huấn luyện lại mô hình trên toàn bộ tập Train với tham số tối ưu
            model.fit(X_train, y_train)

            # Chạy dự đoán thực tế trên cả tập Train và Test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Tính toán chỉ số R2 Score cho 2 tập
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Lưu điểm R2 của tập Test vào Dictionary báo cáo theo tên mô hình
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Hàm giải mã (deserialize) đọc file pickle (.pkl) từ ổ đĩa 
    và khôi phục lại đối tượng Python ban đầu (dùng khi chạy Pipeline Predict).
    """
    try:
        # Mở file ở chế độ Đọc Nhị Phân (Read Binary - "rb") và tải dữ liệu
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)