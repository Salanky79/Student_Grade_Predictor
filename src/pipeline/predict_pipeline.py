import os  # Bổ sung os để sử dụng os.path.join
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    Class quản lý luồng dự đoán (Inference Pipeline):
    Tải các file đã huấn luyện (preprocessor & model) và thực hiện dự đoán trên dữ liệu mới.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Nhận vào dữ liệu dạng DataFrame, chuẩn hóa qua preprocessor, 
        sau đó đưa qua model để trả về kết quả dự đoán.
        """
        try:
            # Khai báo đường dẫn tới các file pkl lưu trong thư mục artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Before Loading")
            # Tải đối tượng model và preprocessor từ ổ đĩa
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Biến đổi (Scaling / Encoding) dữ liệu đầu vào theo quy chuẩn đã train
            data_scaled = preprocessor.transform(features)
            
            # Thực hiện dự đoán kết quả
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class đóng vai trò làm cầu nối (Data Mapper):
    Thu thập từng trường thông tin từ Form/API và chuẩn hóa thành định dạng Pandas DataFrame.
    """
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        # Lưu trữ thông tin người dùng nhập vào các thuộc tính của đối tượng
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Chuyển đổi dữ liệu thuộc tính cá nhân thành một Pandas DataFrame 1 dòng,
        có tên các cột trùng khớp hoàn toàn với tập dữ liệu lúc huấn luyện mô hình.
        """
        try:
            # Tạo Dictionary chứa dữ liệu dưới dạng danh sách (list)
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Chuyển đổi Dictionary thành DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)