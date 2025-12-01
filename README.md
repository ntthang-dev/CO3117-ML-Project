# 🚘 Dự án Học Máy: Dự đoán Giá xe Audi đã qua sử dụng

**(Machine Learning Project - Audi Car Price Prediction)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED)

---

## 📖 Giới thiệu (Introduction)

Chào mừng đến với dự án của chúng mình! 👋

Dự án này được xây dựng nhằm giải quyết bài toán **định giá xe ô tô cũ** một cách tự động. Thay vì phải đoán mò giá trị của một chiếc xe Audi dựa trên kinh nghiệm, hệ thống này sử dụng các thuật toán **Học máy (Machine Learning)** và **Học sâu (Deep Learning)** để phân tích dữ liệu lịch sử và đưa ra mức giá gợi ý chính xác nhất.

**Mục tiêu chính:**

-   📊 Phân tích các yếu tố ảnh hưởng đến giá xe.
-   🤖 Xây dựng mô hình dự đoán giá với độ chính xác cao.
-   🏆 So sánh hiệu quả giữa các thuật toán khác nhau.

---

## 👥 Đội ngũ Thực hiện

| STT | Thành viên                | MSSV    | Vai trò & Nhiệm vụ                              |
| :-: | :------------------------ | :------ | :---------------------------------------------- |
|  1  | 👨‍💻 **Nguyễn Trọng Thắng** | 1915244 | Deep Learning (MLP), Ensemble, Docker, Tổng hợp |
|  2  | 👨‍💻 **Lê Phương Vũ**       | 2313954 | Mô hình SVM, Viết tài liệu                      |
|  3  | 👨‍💻 **Nguyễn Thanh Lộc**   | 2311958  | Hồi quy tuyến tính, vẽ biểu đồ & Đánh giá       |
|  4  | 👨‍💻 **Đặng Quốc Bảo**      | 2210200  | Xử lý dữ liệu, Random Forest                    |

---

## 🔍 Dữ liệu & Thuộc tính (Data & Features)

Mô hình được huấn luyện trên bộ dữ liệu **10,668 chiếc xe Audi**. Các thông tin được sử dụng để "dạy" máy tính bao gồm:

Nguồn Dataset : Dataset sử dụng trong dự án này ("Audi used car listings") được lấy từ [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/audi-used-car-listings)

-   🏎️ **Model:** Dòng xe (A1, A3, Q5, R8...)
-   📅 **Year:** Năm sản xuất
-   ⚙️ **Transmission:** Loại hộp số (Tự động, Sàn, Bán tự động)
-   🛣️ **Mileage:** Số dặm đã đi (Odo)
-   ⛽ **FuelType:** Loại nhiên liệu (Xăng, Dầu, Hybrid)
-   💰 **Tax:** Thuế đường bộ
-   🔥 **MPG:** Mức tiêu thụ nhiên liệu (Dặm/Gallon)
-   🚀 **EngineSize:** Dung tích động cơ

---

## 🧠 Các Mô hình được Áp dụng (Algorithms)

Chúng mình không chỉ dùng một cách, mà thử nghiệm **5 phương pháp** khác nhau để tìm ra "nhà vô địch":

1.  📈 **Linear Regression (Hồi quy tuyến tính):** Mô hình cơ bản, đơn giản nhất.
2.  🌲 **Random Forest (Rừng ngẫu nhiên):** Mô hình mạnh mẽ dựa trên cây quyết định (đã được tinh chỉnh).
3.  📐 **Support Vector Machine (SVM):** Tìm kiếm biên giới hạn tối ưu cho dữ liệu.
4.  🧠 **Multi-layer Perceptron (Deep Learning):** Mạng nơ-ron nhân tạo mô phỏng não bộ.
5.  🤝 **Voting Regressor (Ensemble Learning):** Kỹ thuật **"Hội đồng bỏ phiếu"**, kết hợp sức mạnh của cả 4 mô hình trên để đưa ra kết quả tốt nhất.

---

## 📂 Cấu trúc Dự án

```text
DỰ_ÁN_ML/
│
├── data/                   # 💾 Nơi chứa dữ liệu
│   └── audi.csv            # File dữ liệu gốc
│
├── src/                    # 🧠 Bộ não của chương trình (Mã nguồn)
│   ├── preprocessing.py    # Làm sạch & Chuẩn hóa dữ liệu
│   ├── linear_regression.py# Code chạy Hồi quy tuyến tính
│   ├── svm.py              # Code chạy SVM
│   ├── random_forest.py    # Code chạy Random Forest
│   ├── mlp.py              # Code chạy Deep Learning
│   ├── ensemble.py         # Code chạy Voting Regressor
│   └── utils.py            # Công cụ vẽ biểu đồ & Đánh giá
│
├── main.py                 # 🚀 FILE CHẠY CHÍNH
├── requirements.txt        # Danh sách thư viện (cho pip)
├── environment.yml         # Cấu hình môi trường (cho Conda)
├── Dockerfile              # Cấu hình đóng gói (cho Docker)
└── README.md               # Bạn đang đọc file này <-
```

## 🛠️ Hướng dẫn Cài đặt & Chạy (Installation)

Để đảm bảo chương trình chạy mượt mà trên mọi máy tính, chúng mình cung cấp 3 cách cài đặt. Hãy chọn cách bạn thấy quen thuộc nhất nhé!

### Cách 1: Dành cho người dùng Python cơ bản (Pip) 🐍

Đây là cách nhanh nhất nếu máy bạn đã cài Python.

1. Mở Terminal (hoặc CMD/PowerShell) tại thư mục dự án.

2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

3. Chạy chương trình:

```bash
python main.py
```

### Cách 2: Dành cho người dùng Anaconda/Miniconda 🟢

Cách này giúp quản lý môi trường sạch sẽ hơn.

1. Tạo môi trường ảo từ file cấu hình:

```bash
conda env create -f environment.yml
```

2. Kích hoạt môi trường:

```bash
conda activate audi_price_prediction
```

3. Chạy chương trình:

```bash
python main.py
```

### Cách 3: Dành cho người dùng Docker 🐳

Cách này đảm bảo 100% không lỗi môi trường (khuyên dùng nếu chấm bài trên máy lạ).

1. Xây dựng ảnh (Build Image):

```bash
docker build -t audi-prediction-app .
```

2. Chạy ứng dụng (Run Container):

```bash
docker run --rm audi-prediction-app
```

---

## 📊 Kết quả mong đợi (What to expect)

Khi bạn chạy lệnh `python main.py`, hệ thống sẽ tự động thực hiện các bước sau:

-   **Load dữ liệu:** Đọc file audi.csv.
-   **Tiền xử lý:** Tự động điền dữ liệu thiếu, chuyển đổi chữ thành số (Encoding).
-   **Huấn luyện:** Máy tính sẽ lần lượt "học" từ dữ liệu bằng 5 thuật toán khác nhau.
-   **Đánh giá:** In ra màn hình các chỉ số sai số (RMSE) và độ chính xác (R² Score).
-   **Trực quan hóa:**
    -   Hiện biểu đồ so sánh giá Thực tế vs Dự đoán.
    -   Hiện bảng xếp hạng xem thuật toán nào dự đoán chuẩn nhất.

**Lưu ý:** Quá trình chạy có thể mất từ 30 giây đến 2 phút tùy vào cấu hình máy tính của bạn (do mô hình Deep Learning và SVM cần tính toán nhiều).

---

## 📝 Ghi chú

-   Dữ liệu audi.csv cần phải nằm trong thư mục data/.
-   Kết quả có thể chênh lệch rất nhỏ giữa các lần chạy do tính ngẫu nhiên của thuật toán, nhưng thứ hạng mô hình thường không đổi.

**Cảm ơn các bạn đã quan tâm đến dự án này! ❤️**

---
