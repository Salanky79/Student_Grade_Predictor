# 🎓 Student Exam Performance Predictor

Dự án End-to-End Machine Learning dự đoán điểm thi môn Toán của học sinh dựa trên các yếu tố nhân khẩu học, trình độ học vấn gia đình và điểm số các môn học liên quan.

---

## 📌 Tổng Quan Dự Án

Ứng dụng web được xây dựng bằng **Flask** kết hợp mô hình **Machine Learning** đã qua huấn luyện. Người dùng nhập thông tin đầu vào trực tiếp trên giao diện web hiện đại (Bootstrap 5 Dark Mode) và nhận kết quả dự đoán điểm số theo thời gian thực.

### Các biến đầu vào (Features):
* **Giới tính (Gender)**: Male / Female
* **Chủng tộc / Dân tộc (Race/Ethnicity)**: Group A, B, C, D, E
* **Trình độ học vấn của phụ huynh (Parental Education)**: High School, Some College, Associate's, Bachelor's, Master's, v.v.
* **Chế độ ăn trưa (Lunch Type)**: Standard / Free or Reduced
* **Khóa luyện thi (Test Prep Course)**: None / Completed
* **Điểm Đọc (Reading Score)**: 0 - 100
* **Điểm Viết (Writing Score)**: 0 - 100

---

## 🛠️ Công Nghệ Sử Dụng

* **Ngôn ngữ**: Python 3.x
* **Web Framework**: Flask
* **Machine Learning & Data**: Scikit-Learn, Pandas, NumPy
* **Frontend**: HTML5, CSS3, Bootstrap 5 (Giao diện Dark Mode với Animations)
* **Kiến trúc**: Modular Code (Components & Pipelines)

---

## 📂 Cấu Trúc Thư Mục

```text
Student_Grade_Predictor/
├── artifacts/                  # File dữ liệu và mô hình (.pkl)
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
├── src/                        # Mã nguồn chính
│   ├── components/             # Xử lý dữ liệu & Huấn luyện
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/               # Luồng thực thi dự đoán
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py            # Bắt lỗi tùy chỉnh (Custom Exception)
│   ├── logger.py               # Ghi log hệ thống
│   └── utils.py                # Hàm phụ trợ (Load/Save pkl)
├── templates/                  # Giao diện HTML
│   └── home.html
├── app.py                      # Server Flask
├── requirements.txt            # Thư viện phụ thuộc
└── setup.py                    # Đóng gói dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Khởi tạo môi trường ảo & cài đặt thư viện
```bash
# Khởi tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows Git Bash)
source venv/Scripts/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Tạo Artifacts (Nếu chưa có file model.pkl)
Chạy lần lượt các bước tiền xử lý và huấn luyện mô hình:
```bash
python src/components/data_ingestion.py
python src/components/data_transformation.py
python src/components/model_trainer.py
```

### 3. Khởi chạy Ứng dụng Web
```bash
python app.py
```
Trình duyệt sẽ tự động mở tại địa chỉ: `http://127.0.0.1:5000/`

---

## 🎨 Điểm Nổi Bật Giao Diện
* Tông màu xám đen (Dark Slate) tối ưu mắt nhìn.
* Hiệu ứng chuyển động mượt mà khi tương tác và nhập liệu.
* Điểm dự đoán hiển thị nổi bật kèm hiệu ứng phát sáng (Glow).