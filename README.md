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

## 4. Tổng quan về đề tài

### 4.1. Bối cảnh nghiên cứu

Trong bối cảnh phát triển bền vững và chuyển đổi xanh, việc đánh giá phát thải carbon của sản phẩm ngày càng trở thành yêu cầu quan trọng đối với doanh nghiệp, chuỗi cung ứng và các hoạt động quản lý môi trường. Một trong những chỉ số được sử dụng phổ biến là **Product Carbon Footprint (PCF)**, phản ánh lượng phát thải khí nhà kính phát sinh trong vòng đời của một sản phẩm, thường được quy đổi về đơn vị **kg CO₂e**.

Tuy nhiên, việc tính toán PCF theo phương pháp LCA truyền thống thường đòi hỏi nhiều dữ liệu kiểm kê chi tiết, chuyên môn môi trường và thời gian xử lý. Vì vậy, đề tài này xây dựng hệ thống **EcoPredict Carbon** như một prototype hỗ trợ ước lượng sơ bộ PCF bằng cách kết hợp dữ liệu môi trường, tư duy LCA và các mô hình Machine Learning.

### 4.2. Mục tiêu đề tài

Mục tiêu của đề tài là xây dựng một hệ thống hỗ trợ dự báo và phân tích phát thải carbon của sản phẩm, có khả năng:

* Ước lượng sơ bộ PCF của sản phẩm dựa trên các thông tin đầu vào như năm, quốc gia/khu vực, nhóm ngành, ngành sản phẩm, khối lượng và tỷ trọng vòng đời.
* Phân loại mức phát thải carbon của sản phẩm thành các nhóm **Thấp / Trung bình / Cao**.
* So sánh kết quả PCF của sản phẩm với trung vị ngành, trung vị toàn bộ dữ liệu và dữ liệu tham chiếu OpenPCF.
* Kết hợp phương pháp **LCA bottom-up** thông qua công thức `activity data × emission factor`.
* Mô phỏng kịch bản phát thải trong tương lai bằng mô hình **scenario-based projection**, dựa trên các giả định về năng lượng, vật liệu, vận chuyển và tối ưu chuỗi cung ứng.
* Giải thích các yếu tố ảnh hưởng đến kết quả dự báo thông qua permutation importance và SHAP/XAI.
* Hỗ trợ người dùng đánh giá mức tin cậy của kết quả dựa trên phạm vi dữ liệu tham chiếu và chất lượng đầu vào.

### 4.3. Dữ liệu sử dụng

Hệ thống sử dụng và hợp nhất nhiều nguồn dữ liệu liên quan đến phát thải carbon sản phẩm, bao gồm:

* **Carbon Catalogue**: dữ liệu PCF lịch sử của các sản phẩm, dùng làm nền cho huấn luyện mô hình.
* **OpenPCF by Terralytiq**: dữ liệu hệ số phát thải theo sản phẩm/vật liệu, hỗ trợ tham chiếu PCF theo khối lượng.
* **Open CEDA**: dữ liệu hệ số phát thải theo quốc gia, ngành hoặc khu vực, hỗ trợ mở rộng ngữ cảnh phát thải.
* **Inventory bottom-up do người dùng nhập**: gồm vật liệu, năng lượng, vận chuyển, bao bì và xử lý cuối vòng đời.

Các nguồn dữ liệu này được chuẩn hóa, xử lý thiếu dữ liệu, mã hóa biến phân loại và đưa vào pipeline Machine Learning để phục vụ dự báo, phân loại, benchmark và phân tích kịch bản.

### 4.4. Phương pháp tiếp cận

Đề tài sử dụng cách tiếp cận kết hợp giữa **PCF, LCA và Machine Learning**:

* **PCF** là chỉ số trung tâm của hệ thống, dùng để định lượng dấu chân carbon của sản phẩm.
* **LCA** đóng vai trò là khung phương pháp, giúp tổ chức dữ liệu theo vòng đời sản phẩm như upstream, operations, downstream, transport và end-of-life.
* **Machine Learning** được sử dụng để học quan hệ giữa đặc trưng sản phẩm và PCF, từ đó hỗ trợ dự báo, phân loại và giải thích yếu tố ảnh hưởng.
* **Scenario Projection** được dùng thay cho dự báo chuỗi thời gian ARIMA, vì dữ liệu PCF không tạo thành chuỗi thời gian đủ dài và liên tục. Cách tiếp cận này minh bạch hơn, cho phép mô phỏng các kịch bản như baseline, net zero hoặc pessimistic dựa trên các giả định rõ ràng.
* **SHAP/XAI** được bổ sung để tăng tính giải thích của mô hình, giúp người dùng hiểu yếu tố nào làm tăng hoặc giảm xu hướng phát thải.

### 4.5. Kiến trúc hệ thống

Hệ thống được triển khai bằng Python và Streamlit, gồm các thành phần chính:

1. **Data Processing Layer**
   Tiền xử lý dữ liệu, hợp nhất các nguồn dữ liệu, xử lý missing values, chuẩn hóa feature và xây dựng tập huấn luyện.

2. **Machine Learning Layer**
   Huấn luyện các mô hình phân loại và hồi quy như Random Forest, Extra Trees, Logistic Regression, Ridge và Dummy Baseline. Hệ thống có bổ sung xử lý mất cân bằng lớp, đánh giá bằng F1-macro, balanced accuracy và confusion matrix.

3. **LCA Bottom-up Layer**
   Cho phép người dùng nhập inventory cơ bản để tính PCF theo công thức `activity data × emission factor`.

4. **Scenario Projection Layer**
   Mô phỏng PCF tương lai theo các kịch bản dựa trên các driver như giảm vật liệu, tăng điện tái tạo, cải thiện vận chuyển và tối ưu supplier/geography.

5. **Explainability Layer**
   Cung cấp permutation importance, SHAP plots, local feature impact và sensitivity analysis để giải thích mô hình.

6. **Streamlit UI Layer**
   Giao diện dashboard trực quan gồm các trang: Dự báo, Dữ liệu, LCA/ISO nâng cao, ML & đánh giá và Quy trình.

### 4.6. Ý nghĩa của đề tài

EcoPredict Carbon không nhằm thay thế kiểm kê LCA chính thức hoặc chứng nhận ISO/EPD, mà đóng vai trò là một **prototype hỗ trợ ra quyết định**. Hệ thống giúp người dùng nhanh chóng ước lượng phát thải carbon, so sánh với benchmark ngành, xác định yếu tố ảnh hưởng chính và thử nghiệm các kịch bản giảm phát thải.

Về mặt học thuật, đề tài thể hiện khả năng kết hợp giữa dữ liệu môi trường, phương pháp LCA và Machine Learning trong một hệ thống phân tích có giao diện trực quan. Về mặt ứng dụng, hệ thống có thể được xem như bước đầu để phát triển các công cụ hỗ trợ doanh nghiệp trong việc đánh giá sơ bộ PCF, lập kế hoạch giảm phát thải và chuẩn bị dữ liệu cho các hoạt động LCA chuyên sâu hơn.

## 5. Giới hạn phạm vi

Hệ thống hiện tại là prototype nghiên cứu, vì vậy có một số giới hạn cần lưu ý:

* Kết quả PCF chỉ mang tính ước lượng và tham khảo.
* Dữ liệu đầu vào được tổng hợp từ nhiều nguồn khác nhau, có thể khác biệt về thời gian, phạm vi và mức độ chi tiết.
* Phân loại Low / Medium / High phụ thuộc vào ngưỡng PCF được xây dựng từ dữ liệu huấn luyện.
* Mô phỏng tương lai là kịch bản giả định, không phải dự báo tuyệt đối.
* Hệ thống chưa phải công cụ chứng nhận ISO 14040, ISO 14067 hoặc EPD chính thức.

Do đó, kết quả từ hệ thống nên được sử dụng cho mục đích phân tích sơ bộ, benchmark và hỗ trợ ra quyết định, không thay thế cho kiểm kê LCA đầy đủ có kiểm định chuyên gia.

## 6. Ghi chú học thuật

Nếu metric như accuracy hoặc ROC AUC rất cao, không nên hiểu là mô hình hoàn hảo. Với bài toán này, nhãn Low/Medium/High được xây dựng từ ngưỡng PCF, đồng thời một số đặc trưng có liên quan đến emission factor. Vì vậy cần ưu tiên xem `F1-macro`, `Balanced Accuracy`, `recall_high`, confusion matrix và kiểm thử hold-out theo sản phẩm/quốc gia/thời gian.
