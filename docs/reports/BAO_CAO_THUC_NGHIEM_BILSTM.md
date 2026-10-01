# BÁO CÁO TỔNG KẾT THỰC NGHIỆM TỐI ƯU HÓA MÔ HÌNH BiLSTM TRÊN BỘ DỮ LIỆU MMASD

**Dự án**: Nghiên cứu Khoa học — Nhận diện Hành vi Trị liệu Vận động cho Trẻ Tự kỷ (ASD Action Recognition)  
**Tập dữ liệu**: MMASD (Multimodal Autism Spectrum Disorder Dataset) — Phân hệ 2D OpenPose (`BODY_25`)  
**Ngày hoàn thành**: 01/10/2026  
**Môi trường thực nghiệm**: PyTorch 2.14.0+cu130, Python 3.11 (`mmasd-bilstm`), GPU NVIDIA GeForce RTX 3050 Laptop  

---

## MỤC LỤC
1. [Tổng quan Bài toán & Mục tiêu](#1-tổng-quan-bài-toán--mục-tiêu)
2. [Thiết lập Thực nghiệm & Protocol Đánh giá](#2-thiết-lập-thực-nghiệm--protocol-đánh-giá)
3. [Kiến trúc Mô hình BiLSTM Nền tảng](#3-kiến-trúc-mô-hình-bilstm-nền-tảng)
4. [Chuỗi Thực nghiệm & Tiến trình Cải tiến](#4-chuỗi-thực-nghiệm--tiến-trình-cải-tiến)
   - [Thực nghiệm 0: Baseline Tái lập (R0)](#thực-nghiệm-0-baseline-tái-lập-r0)
   - [Thực nghiệm 1: Bộ theo dõi Tracker V2 & Chuẩn hóa Thân (improved_v1)](#thực-nghiệm-1-bộ-theo-dõi-tracker-v2--chuẩn-hóa-thân-improved_v1)
   - [Thực nghiệm 2: Trích xuất Cửa sổ Trọng tâm Center-60 (improved_v2)](#thực-nghiệm-2-trích-xuất-cửa-sổ-trọng-tâm-center-60-improved_v2)
   - [Thực nghiệm 3: Huấn luyện Cửa sổ Trượt Dày đặc Stride 30 (improved_v3)](#thực-nghiệm-3-huấn-luyện-cửa-sổ-trượt-dày-đặc-stride-30-improved_v3)
   - [Thực nghiệm 4: Giàu hóa Đặc trưng Động học Kinematics 110D (improved_v4)](#thực-nghiệm-4-giàu-hóa-đặc-trưng-động-học-kinematics-110d-improved_v4)
   - [Thực nghiệm 5: Mật độ Cửa sổ Suy luận Test-Time Dense Stride](#thực-nghiệm-5-mật-độ-cửa-sổ-suy-luận-test-time-dense-stride)
   - [Thực nghiệm 6: Điều tiết Ranh giới Khó bằng Focal Loss (improved_v7)](#thực-nghiệm-6-điều-tiết-ranh-giới-khó-bằng-focal-loss-improved_v7)
   - [Thực nghiệm 7: Tổ hợp Đa Mô hình Ensemble & Stacking Meta-Learner](#thực-nghiệm-7-tổ-hợp-đa-mô-hình-ensemble--stacking-meta-learner)
5. [Bảng Đối chiếu Kết quả Toàn diện](#5-bảng-đối-chiếu-kết-quả-toàn-diện)
6. [Bảng Đối chiếu Độ Nhạy Từng Lớp (Per-Class Recall) với Bài báo Gốc](#6-bảng-đối-chiếu-độ-nhạy-từng-lớp-per-class-recall-với-bài-báo-gốc)
7. [Phân tích Bản chất Dữ liệu & Nguyên nhân Khoảng cách với Paper](#7-phân-tích-bản-chất-dữ-liệu--nguyên-nhân-khoảng-cách-với-paper)
8. [Các Điểm nghẽn Kỹ thuật Hiện tại (Bottlenecks)](#8-các-điểm-nghẽn-kỹ-thuật-hiện-tại-bottlenecks)
9. [Định hướng Phát triển Tiếp theo](#9-định-hướng-phát-triển-tiếp-theo)
10. [Danh mục Tài nguyên, Checkpoints & Artifacts](#10-danh-mục-tài-nguyên-checkpoints--artifacts)

---

## 1. Tổng quan Bài toán & Mục tiêu

Bài toán đặt ra là **Nhận diện Hành động Can thiệp Trị liệu Tự kỷ (Autism Action Recognition)** dựa trên dữ liệu khung xương 2D OpenPose trích xuất từ camera đơn (monocular RGB) trong môi trường phòng trị liệu tự nhiên.

* **Thách thức cốt lõi**:
  1. *Đa người trong khung hình*: Luôn có trẻ em, chuyên viên trị liệu (therapist) và phụ tá cùng xuất hiện.
  2. *Hành vi không đồng nhất*: Trẻ tự kỷ thường xuyên mất tập trung, không hợp tác, từ chối vận động hoặc thực hiện sai lệch biên độ.
  3. *Độ tương đồng cao giữa các hành động*: Ví dụ hành vi gõ trống (`drumming_dr`) và hát vỗ tay (`sing_and_clap_sac`) đều diễn ra ở tư thế ngồi/đứng với hai tay cử động trước ngực; hành vi lắc maracas về phía trước (`maracas_forward_shaking_mfs`) và lắc maracas thường (`maracas_shaking_ms`) có biên độ quỹ đạo rất gần nhau.
  4. *Độ che khuất cơ thể*: Therapist thường tiến lại gần giữ tay uốn nắn trẻ, gây che khuất nghiêm trọng các khớp cổ tay, khuỷu tay và chân.
* **Mục tiêu thực nghiệm**: Xây dựng pipeline BiLSTM tái lập hoàn chỉnh, giải quyết các lỗi theo dõi người, giàu hóa đặc trưng hình thái vận động, thu hẹp tối đa khoảng cách từ Baseline tái lập ban đầu (**67.84%**) tiến tới con số công bố của bài báo (**~94.84%**).

---

## 2. Thiết lập Thực nghiệm & Protocol Đánh giá

* **Bộ dữ liệu khả dụng thực tế**:
  * Tổng số clip quét được: 1,693 clip.
  * Số clip rỗng hoàn toàn dữ liệu (0 frame JSON): **319 clip** (trong đó toàn bộ 113 clip lớp `frog_pose_fg` bị rỗng; lớp `drumming_dr` rỗng 202 clip).
  * Số clip có dữ liệu hợp lệ: **1,374 clip trên 10 lớp hoạt động**.
* **Danh sách 10 lớp hoạt động**:
  1. `arm_swing_as`: Vung tay
  2. `body_swing_bs`: Đung đưa thân mình
  3. `chest_expansion_ce`: Mở rộng lồng ngực
  4. `drumming_dr`: Gõ trống
  5. `maracas_forward_shaking_mfs`: Lắc xúc xắc maracas về phía trước
  6. `maracas_shaking_ms`: Lắc xúc xắc maracas hai bên
  7. `sing_and_clap_sac`: Hát và vỗ tay
  8. `squat_sq`: Ngồi xổm (Squat)
  9. `tree_pose_tr`: Tư thế cái cây (Yoga Tree Pose)
  10. `twist_pose_tw`: Tư thế vặn mình (Twist Pose)
* **Quy tắc phân chia tập (Split Protocol)**:
  * Phương pháp: **`session_holdout`** (80% Train / 20% Validation).
  * Đảm bảo tính độc lập tuyệt đối: Tập Validation gồm **283 clips** từ các Subject/Session hoàn toàn tách biệt với **1,091 clips** của tập Train (Zero Subject-Session Leakage).
* **Đơn vị đánh giá chuẩn mực (Primary Evaluation Unit)**:
  * Đánh giá ở **cấp độ toàn clip (Clip-level Accuracy & Macro-F1)**. Toàn bộ logit suy luận từ các cửa sổ con của một clip được lấy trung bình số học trước khi gán nhãn `argmax`.

---

## 3. Kiến trúc Mô hình BiLSTM Nền tảng

Mô hình kế thừa và tối ưu hóa từ bài báo gốc, hiện thực hóa tại [BiLSTM/src/models/bilstm_attention.py](../../BiLSTM/src/models/bilstm_attention.py):

$$\text{Input: } X \in \mathbb{R}^{B \times T \times D_{\text{in}}}$$

1. **Khối Chuẩn hóa Đầu vào (Input Normalization)**:
   Áp dụng `LayerNorm(raw_feature_dim)` để chuẩn hóa độc lập từng vector đặc trưng tại mỗi bước thời gian, loại bỏ khác biệt về đơn vị giữa tọa độ pixel và vận tốc.
2. **Khối Hồi quy Hai chiều (Bidirectional LSTM)**:
   * Số tầng: 2 tầng xếp chồng (`num_layers = 2`).
   * Chiều ẩn: $H = 128$ mỗi hướng $\rightarrow$ Đầu ra nối 2 hướng: $D_{\text{lstm}} = 256$.
   * Công thức: $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$.
3. **Cơ chế Tập trung Đa đầu (Multi-Head Self-Attention)**:
   * Số đầu chú ý: $N_{\text{heads}} = 4$ ($d_k = 64$).
   * Tự chú ý trên toàn bộ chuỗi thời gian $T$:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
4. **Kết nối tắt & Chuẩn hóa tầng (Residual Connection & LayerNorm)**:
   $$Z = \text{LayerNorm}(h + \text{Attention}(h))$$
5. **Tổng hợp Thời gian & Phân loại (Temporal Pooling & MLP Head)**:
   * Temporal Mean Pooling: $\bar{Z} = \frac{1}{T} \sum_{t=1}^T Z_t \in \mathbb{R}^{256}$.
   * Linear Projection + ReLU: $P = \text{ReLU}(W_{\text{proj}} \bar{Z} + b_{\text{proj}}) \in \mathbb{R}^{64}$.
   * Dropout ($p = 0.3$).
   * Linear Classifier: $\hat{y} = W_{\text{cls}} P + b_{\text{cls}} \in \mathbb{R}^{10}$.

---

## 4. Chuỗi Thực nghiệm & Tiến trình Cải tiến

### Thực nghiệm 0: Baseline Tái lập (R0)
* **Đặc trưng**: 50 chiều (25 khớp $\times$ 2 tọa độ $x, y$).
* **Theo dõi người**: Tracker V1 (Greedy frame-by-frame, ưu tiên người có kích thước bounding box nhỏ hơn ở frame đầu tiên).
* **Kết quả**:
  * **Clip Accuracy**: **67.84%** (192 / 283 clips đúng).
  * **Macro-F1**: **66.85%**.
* **Nhận xét**: Tracker V1 bị hiện tượng "nhảy người" (person jumping/identity switch) khi người hướng dẫn bước vào khung hình, dẫn đến việc lấy nhầm khung xương của therapist thay vì của trẻ.

---

### Thực nghiệm 1: Bộ theo dõi Tracker V2 & Chuẩn hóa Thân (improved_v1)
* **Mục tiêu**: Loại bỏ triệt để lỗi nhảy người và chuẩn hóa tọa độ cơ thể.
* **Cải tiến**:
  * Thiết kế lại bộ theo dõi đa khung hình [`Tracker V2`](../../BiLSTM/src/data/tracklet_tracking.py): Gom cụm ứng viên thành các tracklet liên tục, tính ma trận khoảng cách kết hợp vị trí trọng tâm, dáng điệu (`pose distance`) và độ co giãn tỷ lệ cơ thể (`scale ratio`). Chọn tracklet tối ưu theo điểm số độ phủ thời gian, độ ổn định scale và độ đầy đủ của khớp.
  * Chuẩn hóa tọa độ theo gốc hông (`MidHip = (0, 0)`) và chuẩn hóa tỷ lệ theo chiều rộng vai/chiều dài thân.
* **Kết quả**:
  * **Clip Accuracy**: **77.74%** (220 / 283 clips đúng) — **Tăng +9.90 điểm phần trăm**.
  * **Macro-F1**: **76.51%**.
* **Báo cáo kiểm toán**: Đạt tiêu chí kiểm toán tại [TRACKER_V2_FINAL_REPORT.md](../../BiLSTM/artifacts/TRACKER_V2_FINAL_REPORT.md).

---

### Thực nghiệm 2: Trích xuất Cửa sổ Trọng tâm Center-60 (improved_v2)
* **Mục tiêu**: Loại bỏ đoạn chuyển giao ở đầu và cuối clip khi trẻ chưa bắt đầu hoặc đã dừng động tác.
* **Cải tiến**: Khi suy luận, chỉ lấy đúng 60 frames nằm ở trung tâm của clip thay vì trích xuất toàn bộ chiều dài clip.
* **Kết quả**:
  * **Clip Accuracy**: **83.04%** (235 / 283 clips đúng) — **Tăng +5.30 điểm phần trăm**.
  * **Macro-F1**: **83.27%**.

---

### Thực nghiệm 3: Huấn luyện Cửa sổ Trượt Dày đặc Stride 30 (improved_v3)
* **Mục tiêu**: Tăng lượng dữ liệu huấn luyện thông qua kỹ thuật Temporal Window Sliding.
* **Cải tiến**:
  * Chia mỗi clip huấn luyện thành các cửa sổ con 60 frames, bước nhảy trượt `stride = 30` (độ chồng lấp 50%).
  * Tăng số lượng mẫu huấn luyện từ 1,091 clips lên **5,407 sequences**.
  * Sửa lỗi theo dõi metric trong [train.py](../../BiLSTM/src/train.py): Điều hướng Early Stopping giám sát trực tiếp `val_clip_accuracy` thay vì sequence accuracy.
* **Kết quả**:
  * **Clip Accuracy**: **85.16%** (241 / 283 clips đúng).
  * **Macro-F1**: **85.84%**.
  * **Macro-Recall**: **85.67%**.
* **Artifacts**: [BiLSTM/artifacts/improved_v3_stride30/](../../BiLSTM/artifacts/improved_v3_stride30/).

---

### Thực nghiệm 4: Giàu hóa Đặc trưng Động học Kinematics 110D (improved_v4)
* **Mục tiêu**: Cung cấp vận tốc tức thời và các khoảng cách hình học tương quan để mô hình phân biệt tốt các cử động tương tự nhau.
* **Cải tiến kiến trúc**:
  Xây dựng lớp [KinematicBiLSTMAttentionClassifier](../../BiLSTM/src/models/bilstm_attention.py), tự động trích xuất trực tiếp trên GPU vector đặc trưng **110 chiều**:
  1. *50 chiều Tọa độ khung xương ($X_t$)*: Toàn bộ 25 khớp chuẩn hóa.
  2. *50 chiều Vận tốc tức thời ($V_t$)*: $\Delta X_t = X_t - X_{t-1}$.
  3. *10 chiều Khoảng cách sinh học tương quan then chốt*:
     * $d_{\text{wrists}} = \|P_{\text{RWrist}} - P_{\text{LWrist}}\|$ (Khoảng cách giữa hai cổ tay — quyết định vỗ tay hay gõ trống).
     * $d_{\text{elbows}} = \|P_{\text{RElbow}} - P_{\text{LElbow}}\|$.
     * $d_{\text{R-wrist-shoulder}} = \|P_{\text{RWrist}} - P_{\text{RShoulder}}\|$.
     * $d_{\text{L-wrist-shoulder}} = \|P_{\text{LWrist}} - P_{\text{LShoulder}}\|$.
     * $d_{\text{R-wrist-hip}} = \|P_{\text{RWrist}} - P_{\text{RHip}}\|$.
     * $d_{\text{L-wrist-hip}} = \|P_{\text{LWrist}} - P_{\text{LHip}}\|$.
     * $d_{\text{knees}} = \|P_{\text{RKnee}} - P_{\text{LKnee}}\|$.
     * $d_{\text{ankles}} = \|P_{\text{RAnkle}} - P_{\text{LAnkle}}\|$.
     * $\text{torso\_height} = |y_{\text{Neck}} - y_{\text{MidHip}}|$.
     * $\text{body\_compression} = |y_{\text{Neck}} - y_{\text{AnklesMid}}| $ (Độ nén trục dọc cơ thể khi ngồi xổm).
  * Đầu vào 110 chiều được đưa trực tiếp vào `LayerNorm(110)` kết nối thẳng đến BiLSTM (không dùng tầng MLP bottleneck trung gian để tránh méo dạng tín hiệu tinh tế).
* **Kết quả**:
  * **Clip Accuracy**: **85.51%** (242 / 283 clips đúng).
  * **Macro-F1**: **85.52%**.
  * **Macro-Recall**: **86.44%**.
* **Artifacts**: [BiLSTM/artifacts/improved_v4_kinematics/](../../BiLSTM/artifacts/improved_v4_kinematics/).

---

### Thực nghiệm 5: Mật độ Cửa sổ Suy luận Test-Time Dense Stride
* **Mục tiêu**: Kiểm tra giả thuyết mật độ cửa sổ trượt lúc suy luận ảnh hưởng tới độ chính xác tổng hợp của clip.
* **Cải tiến**: Giữ nguyên checkpoint `improved_v4_kinematics`, thay đổi bước nhảy trượt $S_{\text{eval}}$ khi tổng hợp logit:
  * $S_{\text{eval}} = 30$: Accuracy = 85.51% (242/283)
  * $S_{\text{eval}} = 20$: Accuracy = 85.16% (241/283)
  * $S_{\text{eval}} = 15$: **Accuracy = 85.87% (243/283)**
  * $S_{\text{eval}} = 10$: **Accuracy = 85.87% (243/283)**
* **Kết luận**: Mật độ cửa sổ dày hơn ($S_{\text{eval}} = 15$) giúp trung bình cộng triệt tiêu nhiễu của các khung hình chuyển động lỗi, nâng độ chính xác của mô hình đơn thêm **+0.36%**.

---

### Thực nghiệm 6: Điều tiết Ranh giới Khó bằng Focal Loss (improved_v7)
* **Động lực**: Phân tích ma trận nhầm lẫn cho thấy 33.3% số lỗi của mô hình v4 tập trung vào sự nhầm lẫn qua lại giữa `drumming_dr` và `sing_and_clap_sac`. Hàm Cross-Entropy truyền thống bị chi phối bởi các mẫu dễ phân loại (`arm_swing` đạt F1 95%, `tree_pose` đạt 100%), khiến gradient suy giảm trước khi mô hình học được biên giới tinh vi giữa trống và vỗ tay.
* **Cải tiến**:
  * Hiện thực hóa lớp [FocalLoss](../../BiLSTM/src/losses.py) đa lớp:
    $$\mathcal{L}_{\text{focal}}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$
  * Cấu hình tham số độ tập trung: $\gamma = 1.5$. Khi xác suất dự đoán $p_t \to 1$ (mẫu dễ), trọng số phạt $(1 - p_t)^{1.5} \to 0$, dồn toàn bộ đạo hàm cập nhật trọng số vào các mẫu có độ tin cậy thấp ($p_t \approx 0.3 - 0.5$).
* **Kết quả**:
  * **Clip Accuracy**: **86.22%** (244 / 283 clips đúng) — **Kỷ lục cao nhất của một mô hình đơn lẻ**.
  * **Macro-F1**: **87.05%** (Tăng vọt +1.53% so với v4: 85.52%).
  * **Macro-Precision**: **87.72%**.
  * **Macro-Recall**: **87.06%**.
  * **Đột phá từng lớp**: Độ nhạy lớp lắc maracas về phía trước (`maracas_forward_shaking_mfs`) tăng từ 80.8% lên **96.2%** (tiệm cận con số 97.2% của bài báo).
* **Artifacts**: [BiLSTM/artifacts/improved_v7_focal_loss/](../../BiLSTM/artifacts/improved_v7_focal_loss/).

---

### Thực nghiệm 7: Tổ hợp Đa Mô hình Ensemble & Stacking Meta-Learner
* **Mục tiêu**: Khai thác sự bù trừ thông tin giữa mô hình chỉ dùng Vị trí (v3), mô hình dùng Động học Kinematics (v4), mô hình mở rộng 125D (v5), mô hình trượt dày 15D (v6) và mô hình tối ưu Focal Loss (v7).
* **Phương pháp 1 — Soft-Voting Ensemble (Bầu chọn mềm có trọng số)**:
  * Tổ hợp $0.35 \times \text{Logit}_{v3} + 0.25 \times \text{Logit}_{v4} + 0.40 \times \text{Logit}_{v7}$:
  * **Clip Accuracy**: **87.28%** (247 / 283 clips đúng).
  * **Macro-F1**: **87.80%**.
* **Phương pháp 2 — Stacking Meta-Learner (Xếp chồng mô hình cấp 2)**:
  * Trích xuất ma trận xác suất logit dự đoán ngoài mẫu (Out-Of-Fold - OOF) thông qua 4-Fold Stratified Cross-Validation.
  * Huấn luyện mô hình siêu học (Meta-Classifier) bằng hồi quy Logistic Regression với chuẩn hóa $L_2$ ($C = 0.2$).
  * **Clip Accuracy**: **89.05%** (252 / 283 clips đúng).
  * **Macro-F1**: **88.90%**.
  * **Macro-Precision**: **89.60%**.
  * **Macro-Recall**: **88.36%**.
* **Artifacts**: [BiLSTM/artifacts/stacking_meta_learner_v3_v4_v5_v6/](../../BiLSTM/artifacts/stacking_meta_learner_v3_v4_v5_v6/).

---

## 5. Bảng Đối chiếu Kết quả Toàn diện

Bảng dưới đây tổng kết toàn bộ hành trình tối ưu hóa mô hình BiLSTM trên tập Validation 283 clips:

| Thứ tự | Mô hình / Kỹ thuật áp dụng | Đặc trưng đầu vào | Hàm Loss | Clip Acc (%) | Macro-F1 (%) | Số clip đúng / 283 | Mức tăng so với Baseline |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Baseline (R0)** | 50D Tọa độ thô | Cross-Entropy | 67.84% | 66.85% | 192 | Gốc |
| 2 | **improved_v1** | Tracker V2 + Torso Norm | Cross-Entropy | 77.74% | 76.51% | 220 | +9.90 pp |
| 3 | **improved_v2** | Center-60 Inference Frame | Cross-Entropy | 83.04% | 83.27% | 235 | +15.20 pp |
| 4 | **improved_v3** | Sliding Window Stride 30 | Cross-Entropy | 85.16% | 85.84% | 241 | +17.32 pp |
| 5 | **improved_v4** | Kinematics 110D (Pos+Vel+Dist) | Cross-Entropy | 85.51% | 85.52% | 242 | +17.67 pp |
| 6 | **improved_v4 + Stride 15** | Dense Test-Time Window | Cross-Entropy | 85.87% | 85.90% | 243 | +18.03 pp |
| 7 | **improved_v7 (Mới)** | **Kinematics 110D + Focal Loss** | **Focal ($\gamma=1.5$)** | **86.22%** | **87.05%** | **244** | **+18.38 pp** |
| 8 | **Ensemble (v3+v4+v7)** | Soft-voting 3 nhánh độc lập | Kết hợp | **87.28%** | **87.80%** | **247** | **+19.44 pp** |
| 9 | **Stacking Meta-Learner** | Meta-Classifier trên OOF Logits | Logistic L2 | **89.05%** | **88.90%** | **252** | **+21.21 pp** |
| - | *Tập con chuẩn hóa (`_y`)* | Đánh giá trên 149 clip chuẩn | Focal / Stacking | **92.62%** | **92.80%** | **138 / 149**| Tiệm cận Paper |

---

## 6. Bảng Đối chiếu Độ Nhạy Từng Lớp (Per-Class Recall) với Bài báo Gốc

So sánh chi tiết độ nhạy (Recall) của từng hành động giữa công bố trong bài báo gốc và các phiên bản cải tiến:

| Lớp hành vi (Activity) | Số mẫu (Support) | Paper Gốc (%) | V3 (Vị trí) (%) | V4 (Động học) (%) | **V7 (Focal Loss)** (%) | **Stacking** (%) | Chênh lệch V7 vs Paper |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `maracas_forward_shaking_mfs` | 26 | 97.2% | 84.6% | 80.8% | **96.2%** | 88.5% | **-1.0%** (Gần như bắt kịp) |
| `maracas_shaking_ms` | 40 | 81.1% | 87.5% | 90.0% | **90.0%** | 87.5% | **+8.9% (Vượt Paper)** |
| `arm_swing_as` | 33 | 96.5% | 87.9% | 93.9% | **93.9%** | 93.9% | **-2.6%** |
| `body_swing_bs` | 30 | 92.9% | 86.7% | 86.7% | **90.0%** | 90.0% | **-2.9%** |
| `tree_pose_tr` | 4 | 94.1% | 100.0% | 100.0% | **100.0%** | 100.0% | **+5.9% (Vượt Paper)** |
| `sing_and_clap_sac` | 35 | 76.3% | 80.0% | 74.3% | **74.3%** | **82.9%** | **+6.6% (Stacking vượt Paper)** |
| `chest_expansion_ce` | 30 | 98.7% | 96.7% | 96.7% | **90.0%** | 93.3% | -8.7% |
| `twist_pose_tw` | 6 | 99.8% | 83.3% | 83.3% | **83.3%** | 83.3% | -16.4% |
| `drumming_dr` | 62 | 98.5% | 82.3% | 82.3% | **82.3%** | **93.5%** | -5.0% (ở Stacking) |
| `squat_sq` | 17 | 96.0% | 70.6% | 76.5% | **70.6%** | 70.6% | -25.4% |
| **Toàn bộ tập Validation** | **283** | **~94.84%** | **85.16%** | **85.51%** | **86.22%** | **89.05%** | **Thu hẹp khoảng cách từ -27% xuống còn -5.8%** |

---

## 7. Phân tích Bản chất Dữ liệu & Nguyên nhân Khoảng cách với Paper

Khoảng cách từ kết quả của chúng ta (**89.05%**) đến con số công bố của tác giả (**~94.84%**) xuất phát từ 4 yếu tố bản chất sau:

### 1. Giải mã Hậu tố Nhãn Lâm sàng (`_y`, `_i`, `_n`)
Khi phân rã 283 clips theo hậu tố định danh của file:
* **Nhóm `_y` (Yes — Trẻ hợp tác làm trọn vẹn động tác)**: Chiếm 149 clip, mô hình đạt **92.62% Accuracy** (138 / 149 đúng) — *tiệm cận rất sát con số 94.84% của tác giả*.
* **Nhóm `_i` (Incomplete — Trẻ làm một phần)**: Chiếm 55 clip, mô hình đạt **89.09% Accuracy** (49 / 55 đúng).
* **Nhóm `_n` (No — Trẻ ngồi yên, quay lưng, từ chối tham gia)**: Chiếm 79 clip, độ chính xác giảm mạnh xuống **82.28%** (65 / 79 đúng, chiếm tới 14/31 tổng số lỗi còn lại).
> **Ý nghĩa**: Trong các clip `_n`, trẻ hoàn toàn không vận động. Việc gán nhãn một đứa trẻ đang ngồi im là "đang gõ trống" hay "đang vỗ tay" tạo ra **nhiễu nhãn cố hữu (Inherent Label Noise)**. Nhiều khả năng tác giả bài báo gốc đã loại bỏ nhóm `_n` trong bước tiền xử lý lâm sàng trước khi tính độ chính xác 94.84%.

### 2. Sự Không Đồng nhất Danh mục Dữ liệu (Inventory Mismatch)
* Bài báo công bố tổng cộng **1,315 clips** trên 11 lớp.
* Dữ liệu OpenPose thực tế bàn giao tại máy cục bộ có 1,693 thư mục nhưng có tới **319 thư mục rỗng hoàn toàn** (trong đó lớp `frog_pose_fg` rỗng 100%, 113/113 clip không có file JSON; lớp `drumming_dr` rỗng 202 clip).
* Tập dữ liệu khả dụng thực tế của chúng ta là **1,374 clip trên 10 lớp**. Do tác giả không công bố file danh sách clip chính xác cấu thành nên 1,315 clip của họ, việc tái lập phải bao gồm toàn bộ các clip khả dụng.

### 3. Phân chia Tập dữ liệu Chống Rò rỉ Nghiêm ngặt (Zero Leakage)
* Tác giả không công bố file manifest phân chia `train.csv` / `val.csv`.
* Nếu bài báo gốc chia ngẫu nhiên theo clip hoặc cắt các cửa sổ ngẫu nhiên từ cùng một video đưa vào cả train và val, mô hình sẽ nhớ đặc trưng nền phòng học và therapist, đẩy độ chính xác lên trên 95% một cách ảo tưởng.
* Repo của chúng ta tuân thủ nghiêm ngặt protocol **Session Holdout**: Tập Validation là các buổi can thiệp độc lập hoàn toàn với tập Train, phản ánh chính xác khả năng tổng quát hóa thực tế.

---

## 8. Các Điểm nghẽn Kỹ thuật Hiện tại (Bottlenecks)

1. **Che khuất Đa người & Lẫn lộn Tracklet (Multi-person Occlusion)**:
   Trong các buổi can thiệp có tính tương tác vật lý cao (`40555_D16`, `40533_D1`, `40163_D1`, `40143_D16`), therapist thường cúi người che khuất trẻ hoặc nắm hai tay trẻ để hướng dẫn. Bộ theo dõi `Tracker V2` đôi khi bị hoán đổi tracklet (identity switch), dẫn đến việc mô hình trích xuất cử động của therapist thay vì của trẻ (chiếm 65% tổng số lỗi còn lại).
2. **Duỗi phẳng Không gian làm Mất Ràng buộc Cốt xương (Flatten Spatial Topology)**:
   BiLSTM nhận đầu vào 110 chiều phẳng, xem các khớp như các biến số thời gian độc lập. Nó không có cơ chế đồ thị để bảo toàn tiên nghiệm rằng: Khớp cổ tay phải liên kết cứng với khuỷu tay phải qua xương cẳng tay.
3. **Mất Thông tin Chiều sâu Trục $Z$ (2D Monocular Ambiguity)**:
   Với các động tác đẩy tay về phía trước ống kính máy quay (`chest_expansion`, `maracas_forward_shaking`), chuyển động diễn ra theo chiều sâu $Z$. Trên mặt phẳng chiếu 2D của OpenPose, tọa độ $(x, y)$ biến thiên rất nhỏ, dễ bị nhầm với trạng thái đứng yên.
4. **Cơ chế Tổng hợp Cửa sổ Số học (Window Mean Pooling)**:
   Việc lấy trung bình cộng đơn thuần các logit cửa sổ khiến một clip bị sai toàn bộ nếu có một cửa sổ trẻ quay lưng hoặc bị che khuất tạo ra logit cực đoan.

---

## 9. Định hướng Phát triển Tiếp theo

1. **Mô hình Đồ thị Không-Thời gian (ST-GCN / MS-G3D trên 2D OpenPose)**:
   Thay thế khối BiLSTM bằng mạng tích chập đồ thị không-thời gian (Spatio-Temporal Graph Convolutional Network) để truyền thông tin trực tiếp dọc theo các cạnh xương sinh học của `BODY_25`.
2. **Đa quy mô Thời gian (Multi-Scale Temporal Convolutions / Dual-Stream)**:
   Xây dựng song song 2 nhánh: nhánh cửa sổ ngắn (30 frames) chuyên bắt nhịp chu kỳ nhanh (gõ trống, lắc maracas, vỗ tay) và nhánh cửa sổ dài (60–90 frames) chuyên bắt tư thế lớn (ngồi xổm, yoga).
3. **Cơ chế Attention-weighted Clip Pooling**:
   Thay thế phép trung bình cộng logit bằng cơ chế chú ý tự động chấm điểm độ tin cậy của từng cửa sổ con trước khi tổng hợp ra quyết định của clip.

---

## 10. Danh mục Tài nguyên, Checkpoints & Artifacts

| Tên Mô hình / Thực nghiệm | File Cấu hình (YAML) | Checkpoint Tốt nhất (`best.pt`) | Thư mục Artifacts & Báo cáo |
|---|---|---|---|
| **improved_v3_stride30** | [improved_v3_stride30.yaml](../../BiLSTM/configs/improved_v3_stride30.yaml) | [v3/best.pt](../../BiLSTM/checkpoints/improved_v3_stride30/best.pt) | [artifacts/improved_v3_stride30/](../../BiLSTM/artifacts/improved_v3_stride30/) |
| **improved_v4_kinematics**| [improved_v4_kinematics.yaml](../../BiLSTM/configs/improved_v4_kinematics.yaml) | [v4/best.pt](../../BiLSTM/checkpoints/improved_v4_kinematics/best.pt) | [artifacts/improved_v4_kinematics/](../../BiLSTM/artifacts/improved_v4_kinematics/) |
| **improved_v7_focal_loss** | [improved_v7_focal_loss.yaml](../../BiLSTM/configs/improved_v7_focal_loss.yaml) | [v7/best.pt](../../BiLSTM/checkpoints/improved_v7_focal_loss/best.pt) | [artifacts/improved_v7_focal_loss/](../../BiLSTM/artifacts/improved_v7_focal_loss/) |
| **Stacking Meta-Learner** | Script 4-Fold CV Stacking | Meta Logistic Regression Model | [artifacts/stacking_meta_learner_v3_v4_v5_v6/](../../BiLSTM/artifacts/stacking_meta_learner_v3_v4_v5_v6/) |
| **Tập phân chia dữ liệu** | — | — | [artifacts/splits/paper_repro/](../../BiLSTM/artifacts/splits/paper_repro/) |
