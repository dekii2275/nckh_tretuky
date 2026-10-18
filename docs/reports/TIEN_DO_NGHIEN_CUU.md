# Tiến độ nghiên cứu và nhật ký cập nhật

**Cập nhật gần nhất:** 18/10/2026.
**Phạm vi bản ghi:** đối chiếu tài liệu, mã nguồn và artifact; đã chạy audit/khóa pipeline cùng smoke test trên dữ liệu thật, chưa chạy lại toàn bộ thực nghiệm hoặc huấn luyện mô hình.

## 1. Hướng nghiên cứu hiện tại

**Tên đề tài tạm thời:** Nghiên cứu phương pháp đánh giá và hiệu chỉnh chất lượng dữ liệu khung xương của trẻ từ video phục vụ phân tích hành vi.

Mục tiêu đã chốt ngày 14/10/2026: xây dựng pipeline tự động đánh giá, phát hiện và hiệu chỉnh lỗi skeleton của trẻ em, kiểm chứng chủ yếu trên dữ liệu trẻ ASD. Câu hỏi chính là liệu phương pháp có cải thiện độ tin cậy của dữ liệu mà vẫn bảo toàn chuyển động thật hay không.

Trước mắt dùng skeleton **MMASD có sẵn**. Chưa có video trẻ để kiểm chứng bước video → pose extraction. DREAM và DRYAD phục vụ khảo sát/mở rộng, giữ riêng topology, đơn vị và nhãn tác vụ. Nhận dạng hành động là phép kiểm chứng bổ sung; viewer là công cụ kiểm tra và review.

Chi tiết: [kế hoạch nghiên cứu](../plans/KE_HOACH_NGHIEN_CUU_SKELETON_RELIABILITY.md) và [kế hoạch triển khai 7 ngày](../plans/KE_HOACH_7_NGAY_SAN_PHAM_CHUAN_HOA_SKELETON.md).

## 2. Đã làm được những gì?

| Hạng mục | Đầu ra hiện có | Giới hạn cần nhớ |
|---|---|---|
| Khảo sát dữ liệu | Có báo cáo đối chiếu MMASD, DREAM, DRYAD và mapping liên bộ | Chưa có bộ dữ liệu chuẩn hóa mới cho hướng hiệu chỉnh; không gộp nhãn ba bộ |
| Pipeline MMASD | Parser OpenPose BODY_25, các chế độ tracking, preprocessing, tạo sequence/window, train/evaluate BiLSTM + Attention; đã khóa pipeline quan sát v0.1, chuẩn hóa code và tạo snapshot v0.2 | 7/8 thực nghiệm nhất quán; checkpoint/history đối chứng V2 không khớp kết quả cũ. Hash raw data mới chỉ phủ 3 clip smoke |
| Thực nghiệm nhận dạng | Có baseline, các biến thể đặc trưng/cửa sổ/Focal Loss và stacking; số liệu tại mục 3 | Là kết quả trên validation; chưa thay thế đánh giá test độc lập |
| Phân tích lỗi đầu vào | Có thống kê missing joints, confidence và các yếu tố liên quan tới lỗi nhận dạng | Phân tích mô tả, chưa chứng minh quan hệ nhân quả |
| Review 2D/3D MMASD | Có 120 clip / 27.374 frame, join theo frame ID, association OpenPose–ROMP và renderer cơ thể minh họa | Association là heuristic; chưa xác minh đúng danh tính trẻ, đơn vị Z ROMP chưa rõ |
| Nhánh DRYAD/MSG3D | Có data contract, manifest, preprocessing, audit duplicate, các bộ chia fold và kết quả random 10-fold | Kết quả random còn có duplicate giữa train/test; cần đánh giá theo split an toàn với duplicate |
| Khảo sát DREAM | Có tài liệu mapping và script sinh viewer skeleton thân trên, confidence và gaze | Không có nhãn đầy đủ về đúng/sai từng trial; không suy gaze target từ vector |
| Thiết kế nghiên cứu mới | Có mục tiêu, giai đoạn P0–P6, kế hoạch pilot, baseline và benchmark đề xuất | Chưa triển khai pipeline mới và chưa có bảng đánh giá hiệu chỉnh trước/sau |

Tài liệu liên quan: [đối chiếu dữ liệu](BAO_CAO_SO_SANH_MMASD_DRYAD_DREAM.md), [mapping DREAM](BA_DAU_RA_PIPELINE_DREAM_MAPPING.md), [contract DRYAD](../../artifacts/dryad_msg3d/DATA_CONTRACT_v1.md), [rà soát và cập nhật viewer](RA_SOAT_MO_PHONG_3D_VA_CHUAN_HOA_DU_LIEU.md).

## 3. Kết quả đã lưu và cách hiểu

### 3.1. Nhận dạng hành động trên MMASD

Báo cáo thực nghiệm ghi nhận 1.693 clip được quét, 319 clip rỗng và **1.374 clip hợp lệ thuộc 10 lớp**; chia `session_holdout` thành **1.091 train / 283 validation**. Bảng dưới lấy trực tiếp từ các file `validation_results.json` đang lưu, đánh giá ở cấp clip.

| Thực nghiệm | Accuracy | Macro-F1 | Nguồn kết quả |
|---|---:|---:|---|
| Baseline `paper_repro` | 67,84% | 65,75% | [JSON](../../BiLSTM/artifacts/paper_repro/validation_results.json) |
| ROMP-guided `improved_v1` | 60,42% | 56,97% | [JSON](../../BiLSTM/artifacts/improved_v1/validation_results.json) |
| Center-60 `improved_v2` | 81,63% | 79,88% | [JSON](../../BiLSTM/artifacts/improved_v2_center60/validation_results.json) |
| Cửa sổ stride 30, `improved_v3` | 85,16% | 85,84% | [JSON](../../BiLSTM/artifacts/improved_v3_stride30/validation_results.json) |
| Đặc trưng động học, `improved_v4` | 85,51% | 85,52% | [JSON](../../BiLSTM/artifacts/improved_v4_kinematics/validation_results.json) |
| Focal Loss, `improved_v7` | **86,22%** | **87,05%** | [JSON](../../BiLSTM/artifacts/improved_v7_focal_loss/validation_results.json) |
| Stacking các mô hình v3/v4/v5/v6 | **89,05%** | **88,90%** | [JSON](../../BiLSTM/artifacts/stacking_meta_learner_v3_v4_v5_v6/validation_results.json) |

Mô hình đơn v7 đúng 244/283 clip; stacking đúng 252/283 clip. Đây là kết quả thực nghiệm nhận dạng đã lưu, chưa phải bằng chứng hiệu quả của pipeline hiệu chỉnh mới. Chưa có đánh giá trên test độc lập được ghi nhận trong trang này.

**Kết quả đối chiếu ngày 18/10:** đã tính lại metric từ prediction và đối chiếu history/config/checkpoint, xác nhận các giá trị baseline/v1/v2 trong bảng. Những số 66,85%, 77,74% và 83,04% trong [báo cáo cũ](BAO_CAO_THUC_NGHIEM_BILSTM.md) chưa có bộ bằng chứng gốc tương ứng; báo cáo đã được thêm lưu ý. `improved_v1` dùng `romp_guided`, không phải Tracker V2. Checkpoint/history đối chứng V2 hiện mang experiment `improved_v5_kinematics`, không khớp kết quả đối chứng V2 cũ. Chi tiết và giới hạn: [báo cáo rà soát](RA_SOAT_VA_KHOA_PIPELINE_MMASD.md).

**Split:** không trùng clip/nhóm subject–session, nhưng có 13 subject ID chung giữa train và validation; không gọi là subject holdout. Stacking khớp metric từ prediction đã lưu, chưa xác minh quy trình OOF/meta-model.

### 3.2. Dấu hiệu liên quan giữa chất lượng đầu vào và lỗi nhận dạng

[Phân tích S1](../../BiLSTM/artifacts/diagnostics/error_factor_analysis.md) trên 275 clip validation có prediction:

| Nhóm | Số clip | Tỷ lệ thiếu khớp trung bình |
|---|---:|---:|
| Nhận dạng đúng | 204 | 8,3% |
| Nhận dạng sai | 71 | 20,6% |

Có 8 clip ngắn không có prediction nên không tham gia so sánh. Đây là liên hệ mô tả chưa điều chỉnh theo activity, độ dài hoặc subject; là động lực nghiên cứu chất lượng skeleton, chưa chứng minh thiếu khớp gây sai nhận dạng.

### 3.3. Công cụ review 2D/3D

[Artifact audit viewer](../../BiLSTM/artifacts/tracking_review_3d/mapping_audit.json) ghi **120 clip / 27.374 frame**: 24.306 frame matched, 3.006 unmatched và 62 ambiguous; không còn selected ROMP index vượt phạm vi. Các trạng thái này đo kết quả ghép người theo heuristic, không đo tỷ lệ chọn đúng trẻ.

Viewer đã được sửa mapping, góc nhìn, bề mặt minh họa và fallback hiển thị. Báo cáo viewer có ghi các kiểm tra renderer/mapping/browser của đợt triển khai đó; ngày 18/10 chỉ đối chiếu artifact, không chạy lại các kiểm tra này. Bề mặt là mesh minh họa từ khớp, không phải mesh SMPL được phục hồi. Đơn vị và provenance của nguồn ROMP 3D còn cần xác minh.

[Audit nhãn tracking](../../BiLSTM/artifacts/diagnostics/tracking_audit_summary.md) hiện ghi **0/120 clip được người đánh giá**. Chưa thể công bố tỷ lệ chọn đúng trẻ hoặc tỷ lệ identity switch đã xác minh.

### 3.4. Thực nghiệm DRYAD riêng

[Kết quả round 5](../../artifacts/dryad_msg3d/experiments/round5_random_10fold/aggregate_metrics.json), protocol `random_reproduction_like`, 10 fold: accuracy trung bình **91,75% ± 6,24 điểm phần trăm** (độ lệch chuẩn mẫu giữa các fold); trên tập chỉ gồm bản gốc, không augmentation, **84,00% ± 13,50 điểm phần trăm**. Audit ghi có duplicate hash trùng giữa train/test ở một số fold. Vì vậy đây là kết quả thăm dò của protocol đó; chưa đại diện cho đánh giá an toàn với duplicate và không gộp vào kết quả nhận dạng hoạt động MMASD.

## 4. Giai đoạn hiện tại và việc tiếp theo

**Đang ở giai đoạn P0: đã khóa pipeline quan sát, tiếp tục chọn pilot và chốt protocol đánh giá.** Có nền tảng mã nguồn, phân tích ban đầu và viewer; các đầu ra P1–P5 theo kế hoạch mới chưa hoàn tất. Các thư mục ứng dụng và artifact `standardization_v0_1` dự kiến trong kế hoạch chưa được tạo.

- [x] Có pipeline và kết quả nhận dạng MMASD làm nền tảng.
- [x] Có khảo sát ba dataset, phân tích lỗi đầu vào ban đầu và viewer review 2D/3D.
- [x] Chốt mục tiêu nghiên cứu và lập kế hoạch phát hiện–hiệu chỉnh skeleton.
- [x] Rà soát parser/tracker/preprocessing; đối chiếu artifact, khóa cấu hình quan sát và lưu hash v0.1. Các bằng chứng lịch sử chưa khớp được gắn cờ trong báo cáo.
- [ ] Chọn pilot 20–50 clip, khóa protocol và tách tuning/evaluation theo nhóm; bổ sung hash raw data toàn pilot.
- [ ] Hoàn thành schema v0.1 và adapter xuất/đọc lại dữ liệu có raw/processed, confidence, masks và provenance.
- [ ] Xuất bảng chất lượng và sự kiện cảnh báo theo clip/frame/joint.
- [ ] Triển khai baseline: raw, nội suy ngắn, nội suy + filter; thử sửa outlier có điều kiện.
- [ ] Tạo benchmark missing/spike có kiểm soát, giữ reference, mask, severity và seed.
- [ ] Đo detection, recovery, coverage, bảo toàn chuyển động; so trước/sau, ablation và failure cases.
- [ ] Hoàn thiện luồng viewer so sánh raw/processed và lưu/import review có phiên bản.
- [ ] Khi đủ provenance, kiểm chứng bổ sung bằng nhận dạng hành động trên cùng protocol.

Ưu tiên kế tiếp là khóa pilot và protocol, kiểm tra frame nguồn, sau đó tạo bảng chất lượng cùng baseline hiệu chỉnh bằng [cấu hình v0.2 hiện tại](../../BiLSTM/configs/skeleton_audit_v0_2.yaml). Khi có video phù hợp, bổ sung kiểm chứng trích xuất pose và annotation RGB; đây là phần mở rộng chưa thực hiện.

## 5. Cách cập nhật sau mỗi đợt làm việc

Sau mỗi đợt có thay đổi về tiến độ hoặc kết quả:

1. Sửa ngày cập nhật, trạng thái và checklist trong trang này cho đúng đầu ra thực tế.
2. Bổ sung kết quả kèm nguồn artifact, dữ liệu/split, config và phạm vi đánh giá; tách kết quả đã chạy khỏi phương án dự kiến.
3. Ghi kiểm tra đã chạy và kết quả; nếu chỉ đọc lại artifact thì nói rõ, không ghi là vừa kiểm thử hoặc huấn luyện lại.
4. Thêm một mục nhật ký bên dưới, giữ lịch sử các mốc. Nếu đính chính số liệu cũ, ghi lý do và nguồn mới.
5. Đồng bộ phần tiến độ ở README khi có mốc mới. Chi tiết thực nghiệm dài để trong báo cáo riêng rồi liên kết tới đây.

### Nhật ký

#### 18/10/2026 — Lập bản tổng hợp tiến độ ban đầu

- **Đã làm:** đối chiếu hướng nghiên cứu chốt ngày 14/10 với báo cáo, config và artifact; lập trang tiến độ và thêm liên kết trong README/chỉ mục tài liệu.
- **Kết quả:** ghi nhận nền tảng MMASD, DRYAD, DREAM và viewer; xác định pipeline hiệu chỉnh mới chưa triển khai; gắn cờ các số liệu cũ cần đối chiếu nguồn.
- **Kiểm chứng:** đọc các file kết quả JSON, báo cáo diagnostic/tracking và audit viewer đang lưu; kiểm tra sự tồn tại của các thư mục đầu ra dự kiến. `make check` đạt với 114 file Python và 14 README/tài liệu, các liên kết cục bộ hợp lệ; `git diff --check` đạt. Không huấn luyện lại hoặc chạy benchmark mới.
- **Việc tiếp theo:** khóa pilot/protocol, đối chiếu provenance rồi triển khai schema và bảng chất lượng ban đầu.

#### 18/10/2026 — Hoàn thành bước 1: rà soát và khóa pipeline quan sát

- **Đã làm:** lập sơ đồ parser → tracker → preprocessing → sampling/model; lưu cấu hình độc lập OpenPose 2D + Tracklet V2, helper quan sát trước tiền xử lý và snapshot hash. Thêm lưu ý đối chiếu vào báo cáo lịch sử.
- **Kết quả và bằng chứng:** [báo cáo rà soát](RA_SOAT_VA_KHOA_PIPELINE_MMASD.md), [pipeline lock](../../BiLSTM/artifacts/pipeline_audit_v0_1/pipeline_lock.json), [đối chiếu thực nghiệm](../../BiLSTM/artifacts/pipeline_audit_v0_1/experiment_reconciliation.json), [smoke test](../../BiLSTM/artifacts/pipeline_audit_v0_1/smoke_results.json). 7/8 thực nghiệm nhất quán; checkpoint/history đối chứng V2 không khớp. Split cũ còn 13 subject ID chung giữa train/validation.
- **Kiểm chứng:** 22 kiểm thử parser/tracking/preprocessing/sequence/reproducibility đạt; 3 clip thật / 470 frame cho kết quả xác định và khớp loader tham chiếu. Verify 545 file đạt, kiểm tra âm tính phát hiện cấu hình/hash bị đổi; `make check` đạt (115 file Python, 15 tài liệu), `git diff --check` đạt. Không huấn luyện lại hoặc suy luận lại mô hình.
- **Còn thiếu / việc tiếp theo:** chọn pilot 20–50 clip và chốt nhóm tuning/evaluation, hash toàn pilot, kiểm tra frame nguồn rồi xuất bảng chất lượng. Số liệu lịch sử thiếu bằng chứng gốc và checkpoint đối chứng V2 cũ vẫn chưa khôi phục được.

#### 18/10/2026 — Chuẩn hóa code và tách nhánh nghiên cứu

- **Đã làm:** tạo nhánh `research/skeleton-reliability`; tách hàm/type hints/lint/format script audit; thêm 13 kiểm thử độc lập dữ liệu, lệnh Makefile, tài liệu phát triển và workflow GitHub Actions. Giữ snapshot/script v0.1, tạo v0.2 cho bản code chuẩn hóa.
- **Kết quả và bằng chứng:** [quy trình phát triển](../DEVELOPMENT.md), [config v0.2](../../BiLSTM/configs/skeleton_audit_v0_2.yaml), [snapshot v0.2](../../BiLSTM/artifacts/pipeline_audit_v0_2/pipeline_lock.json), [kiểm thử audit](../../BiLSTM/tests/test_pipeline_lock.py). Kết quả khảo sát và cảnh báo lịch sử không thay đổi sau refactor.
- **Kiểm chứng:** bộ kiểm thử MMASD + DRYAD có 97 kiểm thử đạt; Ruff lint/format đạt; chạy audit v0.2 trên 3 clip / 470 frame và verify 545 file đạt. So sánh v0.1/v0.2 xác nhận kết quả smoke và đối chiếu không đổi. Bản checkout sạch từ staging chạy `make check` và 13 kiểm thử audit đạt khi không có dữ liệu/checkpoint/history. Trạng thái GitHub Actions được theo dõi riêng sau push.
- **Việc tiếp theo:** tiếp tục chọn pilot và đo chất lượng trên nhánh nghiên cứu; cập nhật cùng nhật ký sau mỗi mốc.

Mẫu thêm mốc mới:

```markdown
#### DD/MM/YYYY — Tên mốc công việc

- **Đã làm:** ...
- **Kết quả và bằng chứng:** ... (liên kết báo cáo/artifact, dữ liệu/split/config)
- **Kiểm chứng:** ... (lệnh hoặc kiểm tra, kết quả, giới hạn)
- **Còn thiếu / việc tiếp theo:** ...
```
