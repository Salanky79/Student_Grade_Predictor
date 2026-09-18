import os
import sys
from dataclasses import dataclass

# Import các thuật toán Regression
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Import các tiện ích tùy chỉnh từ thư mục src
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# Class cấu hình: Tự động khởi tạo đường dẫn đầu ra cho file model đã huấn luyện
@dataclass
class ModelTrainerConfig:
    # File mô hình hoàn chỉnh sẽ được lưu tại: artifacts/model.pkl
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        # Khởi tạo đối tượng cấu hình
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Hàm nhận vào dữ liệu Train/Test dạng NumPy Array, tiến hành thử nghiệm
        nhiều mô hình + tinh chỉnh tham số (Hyperparameter Tuning), chọn ra mô hình
        tốt nhất và lưu lại dưới dạng file pickle (.pkl).
        """
        try:
            logging.info("Đang tách tập Train và Test thành Features (X) và Target (y)")
            
            # Tách mảng NumPy: Toàn bộ cột đầu làm X, cột cuối cùng làm y
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Danh sách các thuật toán Machine Learning đưa vào đánh giá
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Lưới tham số (Hyperparameters) để tinh chỉnh cho từng mô hình
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # Linear Regression dùng thông số mặc định
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Gọi hàm evaluate_models từ src/utils.py để GridSearch và tính điểm R2 cho từng model
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            
            # Lấy ra điểm R2 cao nhất từ Dictionary kết quả
            best_model_score = max(sorted(model_report.values()))

            # Tìm tên mô hình tương ứng với điểm R2 cao nhất đó
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Lấy đối tượng mô hình đã được huấn luyện tốt nhất
            best_model = models[best_model_name]

            # Thiết lập ngưỡng đánh giá: Nếu R2 < 0.6 thì coi như không có mô hình nào đạt chuẩn
            if best_model_score < 0.6:
                raise CustomException("Không tìm thấy mô hình đủ tiêu chuẩn (R2 Score < 0.6)")
                
            logging.info(f"Đã tìm thấy mô hình tốt nhất: {best_model_name} với R2 Score = {best_model_score}")

            # Lưu mô hình tốt nhất vào đĩa cứng (artifacts/model.pkl)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Chạy dự đoán thực tế trên tập Test để tính toán và trả về R2 Score cuối cùng
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)