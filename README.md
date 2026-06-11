# EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm

EcoPredict Carbon là prototype nghiên cứu hỗ trợ ước lượng Product Carbon Footprint (PCF), benchmark theo ngành, giải thích yếu tố ảnh hưởng bằng Machine Learning và mô phỏng kịch bản phát thải theo định hướng LCA/ISO. Hệ thống dùng để hỗ trợ phân tích sơ bộ, không thay thế kiểm kê LCA chính thức hoặc chứng nhận ISO/EPD.

## 1. File chính

- `app.py`: giao diện Streamlit.
- `carbon_utils.py`: xử lý dữ liệu, feature engineering, model pipeline, LCA bottom-up, OOD/confidence.
- `scenario_projection.py`: mô phỏng kịch bản tham số, không dùng ARIMA.
- `train_advanced_models.py`: huấn luyện classification/regression, class imbalance handling, SHAP, sensitivity, lưu model.
- `imbalance_handler.py`: phân phối lớp, class weights, SMOTE/fallback, diagnostics lớp High.
- `model_interpretation.py`: helper SHAP/XAI.
- `generate_shap_explanations.py`: sinh lại SHAP plots từ model package.
- `hyperparameter_tuning.py`: GridSearchCV dùng `f1_macro` và SMOTE pipeline cho classification.
- `sensitivity_analysis.py`: tornado chart, heatmap 2 chiều, scenario sensitivity.
- `tests/`: unit tests cho core functions.
- `requirements.txt`: thư viện cần cài.
- `carbon_catalogue.csv`, `data/`: dữ liệu đầu vào.
- `outputs/`: metric, hình, bảng, model đã train.

## 2. Chạy trên Windows PowerShell

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Nếu môi trường đã có `.venv`, có thể chạy nhanh:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 3. Lệnh bổ sung

Sinh lại model và toàn bộ output:

```powershell
python train_advanced_models.py
```

Sinh lại SHAP plots:

```powershell
python generate_shap_explanations.py
```

Chạy GridSearchCV:

```powershell
python hyperparameter_tuning.py
```

Sinh sensitivity plots:

```powershell
python sensitivity_analysis.py
```

Chạy unit tests:

```powershell
python -m pytest tests -v --cov=carbon_utils --cov=scenario_projection --cov-report=term-missing
```

## 4. Nâng cấp đã bổ sung theo góp ý nghiên cứu

### 4.1 Class imbalance handling

Bản mới thêm `imbalance_handler.py` và cập nhật pipeline phân loại:

- Có class distribution report.
- Có balanced class weights.
- Có thể dùng SMOTE trong pipeline `preprocessor -> SMOTE -> model` nếu đã cài `imbalanced-learn`.
- Metric có thêm `recall_high`, `f1_high`, `high_class_warning` để phát hiện mô hình bỏ sót lớp phát thải cao.
- Chọn mô hình theo `f1_macro` và `balanced_accuracy`, không chỉ theo `accuracy`.

### 4.2 SHAP/XAI

Bản mới giữ permutation importance và bổ sung SHAP:

- SHAP feature ranking.
- SHAP beeswarm.
- SHAP waterfall cho một mẫu.
- SHAP dependence plot.
- Bảng `shap_feature_importance.csv`.

### 4.3 Sensitivity analysis

Bản mới thêm:

- Tornado chart: tác động của ±20% emission factor lên PCF.
- 2-way heatmap: vật liệu và năng lượng thay đổi đồng thời.
- Scenario sensitivity: so sánh baseline, net zero và pessimistic.

### 4.4 Scenario projection thay ARIMA

Hệ thống không dùng ARIMA vì dữ liệu PCF không phải chuỗi thời gian liên tục đủ dài. Phần tương lai được mô hình hóa bằng scenario-based parametric projection theo các driver vòng đời:

- Upstream/material.
- Operations/energy.
- Downstream/use.
- Transport/logistics.
- End-of-life.

## 5. Phạm vi sử dụng đúng

Phù hợp để:

- Ước lượng sơ bộ PCF.
- Benchmark sản phẩm với trung vị ngành.
- Giải thích yếu tố ảnh hưởng.
- Mô phỏng kịch bản giảm phát thải.
- Hỗ trợ học thuật và ra quyết định sơ bộ.

Chưa phù hợp để:

- Chứng nhận ISO/EPD chính thức.
- Khai báo ESG/green claims có giá trị pháp lý.
- Thay thế LCA chính thức với physical data collection và critical review.

## 6. Ghi chú học thuật

Nếu metric như accuracy hoặc ROC AUC rất cao, không nên hiểu là mô hình hoàn hảo. Với bài toán này, nhãn Low/Medium/High được xây dựng từ ngưỡng PCF, đồng thời một số đặc trưng có liên quan đến emission factor. Vì vậy cần ưu tiên xem `F1-macro`, `Balanced Accuracy`, `recall_high`, confusion matrix và kiểm thử hold-out theo sản phẩm/quốc gia/thời gian.
