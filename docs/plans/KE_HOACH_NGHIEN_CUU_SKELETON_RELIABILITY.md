# Kế hoạch nghiên cứu: Skeleton Reliability và Robust Action Recognition

**Dự án:** NCKH Trẻ tự kỷ – Skeleton-based Autism Action Recognition  
**Ngày lập:** 01/10/2026  
**Trạng thái:** Kế hoạch nghiên cứu; chưa thay mô hình hoặc pipeline huấn luyện.

## 1. Mục tiêu và phạm vi

Hướng nghiên cứu hiện tại là **Skeleton Reliability Analysis and Robust Skeleton-based Action Recognition**. Mục tiêu là đo các bất ổn của skeleton ước lượng, kiểm tra quan hệ của chúng với lỗi nhận dạng hoạt động trên MMASD, tạo benchmark corruption dựa trên lỗi quan sát được, rồi mới thử dùng thông tin reliability để giảm mức suy giảm hiệu năng.

Chuỗi công việc ưu tiên:

```text
AUDIT REPOSITORY → MEASURE QUALITY → ANALYZE ERRORS
                 → BUILD CORRUPTION BENCHMARK → TEST ROBUSTNESS
                 → IMPROVE MODEL (khi đã có bằng chứng)
```

Không gộp nhãn của MMASD, DREAM và DRYAD thành một bài toán phân loại. Không xem skeleton, confidence hay chỉ số heuristic là ground truth về tư thế hoặc danh tính người.

## 2. Bối cảnh dữ liệu và bằng chứng hiện có

| Dataset | Biểu diễn và nhãn | Vai trò trong hướng nghiên cứu |
|---|---|---|
| **MMASD** | OpenPose BODY_25 2D, nhiều người mỗi frame; 10 lớp hoạt động có OpenPose clip khả dụng trong pipeline hiện tại. | Dataset chính: audit tracking/pose, liên hệ quality với lỗi action recognition, corruption và robustness. |
| **DREAM** | Public segment JSON với 10 upper-body joints 3D, confidence, giá trị `null`, task JA/IM/TT và metadata; không có trial-level correctness đầy đủ. | So sánh phân bố confidence, missingness và độ ổn định theo thời gian; không dùng JA/IM/TT làm nhãn MMASD. |
| **DRYAD** | Kinect v2, 25 joints 3D theo gait cycle; nhãn ASD/TD. Tài liệu cục bộ ghi nhận 100 bản gốc và 700 augmentation, cùng sentinel `100` chưa rõ ngữ nghĩa. | Kiểm tra consistency và outlier trên cảm biến 3D khác; giữ riêng đơn vị subject và augmentation, không coi nhãn ASD/TD là activity. |

Nguồn cục bộ: [đối chiếu ba dataset](../reports/BAO_CAO_SO_SANH_MMASD_DRYAD_DREAM.md), [mapping DREAM](../reports/BA_DAU_RA_PIPELINE_DREAM_MAPPING.md), [contract DRYAD](../../artifacts/dryad_msg3d/DATA_CONTRACT_v1.md). Các con số và trạng thái dữ liệu phải được kiểm tra lại trên bản đang dùng khi chạy audit; báo cáo tổng hợp có thể phản ánh snapshot khác.

### Nền tảng MMASD cần bảo toàn

- [Báo cáo BiLSTM](../reports/BAO_CAO_THUC_NGHIEM_BILSTM.md) ghi 1.374 clip OpenPose hợp lệ, chia `session_holdout` thành 1.091 train và 283 validation clip. Đây là **validation**, không gọi là test độc lập.
- Theo báo cáo đó, clip accuracy lần lượt là 67,84% (R0, 50D), 77,74% (`improved_v1`), 83,04% (center-60), 85,16% (stride 30), 85,51% (kinematics 110D), 86,22% (Focal Loss), 87,28% (soft voting), 89,05% (stacking). Đây là kết quả đã báo cáo trên protocol cụ thể, không phải ước lượng hiệu quả nhân quả của từng thay đổi.
- Chênh lệch 67,84% → 77,74% là động lực nghiên cứu input quality. **Cần đối chiếu provenance trước khi quy mức tăng này cho Tracker V2:** báo cáo gọi `improved_v1` là Tracker V2 + normalization, còn [config `improved_v1`](../../BiLSTM/configs/improved_v1.yaml) hiện ghi `tracking.mode: romp_guided`. [Báo cáo V1/V2 riêng](../../BiLSTM/artifacts/TRACKER_V2_FINAL_REPORT.md) là một so sánh có protocol khác.
- [Phân tích lỗi S1 hiện có](../../BiLSTM/artifacts/diagnostics/error_factor_analysis.md) đã so sánh 204 clip đúng và 71 clip sai trong 275 clip validation được giữ lại; 8 clip ngắn không có prediction. Missing-joint rate trung bình lần lượt khoảng 8,3% và 20,6%; đây là liên hệ mô tả chưa điều chỉnh cho class, độ dài clip hoặc subject. Công việc mới phải **kế thừa và kiểm tra lại**, không trình bày như một phát hiện hoàn toàn mới.
- [Audit nhãn tracking](../../BiLSTM/artifacts/diagnostics/tracking_audit_summary.md) hiện có **0/120 clip được người đánh giá**. Vì vậy chưa thể định lượng tỷ lệ identity switch hay khẳng định skeleton được chọn là trẻ.

## 3. Câu hỏi nghiên cứu

1. **RQ1 — Reliability profile:** MMASD, DREAM và DRYAD có những dạng missing, confidence thấp, temporal instability, geometry inconsistency hoặc tracking ambiguity nào, ở mức độ bao nhiêu?
2. **RQ2 — Liên hệ với lỗi dự đoán:** Trên MMASD, các clip được dự đoán sai có quality profile khác clip đúng hay không, sau khi xét class, độ dài clip và nhóm subject/session?
3. **RQ3 — Độ nhạy với corruption:** Những lỗi nào, ở mức độ nào, làm giảm clip accuracy, macro-F1 và per-class recall mạnh nhất?
4. **RQ4 — Dùng reliability để tăng robustness:** Confidence, validity mask hoặc quality indicators có giúp giảm robustness drop mà vẫn giữ hiệu năng clean hợp lý không?

## 4. Bản đồ pipeline cần xác nhận trong đợt audit đầu

```text
OpenPose JSON (`people[*].pose_keypoints_2d`, 25 × [x,y,c])
  → parse_clip / parse_frame              BiLSTM/src/data/openpose_parser.py
  → select_person_track                    BiLSTM/src/data/tracking.py
      ├─ legacy                            BiLSTM/src/data/child_tracking.py
      ├─ tracklet_v2                       BiLSTM/src/data/tracklet_tracking.py
      └─ romp_guided                       BiLSTM/src/data/romp_guided_tracking.py
  → preprocess_clip                        BiLSTM/src/data/preprocessing.py
  → SequenceBuilder / window extraction    BiLSTM/src/data/sequence.py
  → dataset / model inference              BiLSTM/src/data/dataset.py,
                                           BiLSTM/src/models/bilstm_attention.py
  → clip prediction / aggregation          BiLSTM/src/evaluate.py,
                                           BiLSTM/src/window_study.py và nhánh
                                           đánh giá tương ứng từng experiment
```

Đã thấy trong code: parser giữ `[x,y,confidence]`; preprocessing tạo `valid_mask` từ confidence, finite coordinates và `(0,0)`, rồi có thể điền missing, chuẩn hóa theo MidHip/vai với fallback thân, và tùy config mới smoothing. `SkeletonClipDataset` đưa **tọa độ đã xử lý** vào model; `valid_mask` và metadata preprocessing chỉ được trả ra khi `return_debug=True`. Các nhánh window cần kiểm tra chính xác cách cộng logit ở cấp clip; [window study](../../BiLSTM/src/window_study.py) dùng mean logits. Không suy rằng mọi checkpoint dùng cùng tracking, sequence mode hay aggregation nếu chưa đối chiếu config và artifact.

**Đầu ra audit đợt đầu:** báo cáo `current_pipeline_map.md` với file/hàm/config thực tế của từng checkpoint được chọn; bảng thông tin còn giữ/mất theo từng bước; danh sách artifact có thể join bằng `clip_id`; danh sách metric đo được và metric cần annotation/RGB; các bất nhất giữa báo cáo và config đã được giải quyết hoặc gắn cờ.

## 5. Kế hoạch thực hiện theo giai đoạn

| Giai đoạn | Việc làm chính | Đầu ra và điều kiện hoàn tất |
|---|---|---|
| **P0 — Audit và khóa protocol** | Kiểm tra manifest/split, OpenPose JSON, ba kiểu tracking, zero/missing, interpolation, smoothing, normalization fallback, resampling/windowing, prediction export và checkpoint/config provenance. Xác định một model/checkpoint, split, đơn vị clip và tập clip làm phân tích chính; giữ các model khác làm sensitivity analysis. | Pipeline map có dẫn tới file/hàm/config; bảng provenance; quy tắc join và số clip khớp/thiếu; không sửa training pipeline. |
| **P1 — Chuẩn hóa dữ liệu để audit** | Tạo schema chung ở mức sequence/frame/joint nhưng giữ `dataset_id`, topology và đơn vị của từng bộ; tách raw, selected, preprocessed và model-input. Ghi provenance cho missing, interpolation, padding và frame gốc. | Loader đọc được mẫu kiểm tra từ cả ba bộ; thống kê số record/frame/joint và lý do loại/quarantine. Không ép chung số joint hay label space. |
| **P2 — Skeleton Quality Audit** | Tính missingness, confidence, jitter, geometry và tracking indicators ở cấp frame/joint/clip. Kiểm tra phân bố trước khi đặt ngưỡng; lưu cả metric liên tục và cờ heuristic. | `skeleton_quality.csv` và bảng theo joint/frame có định nghĩa, đơn vị, NA rule, version/config; audit một số mẫu bằng đồ thị hoặc review thủ công. |
| **P3 — Quality vs recognition error** | Join quality của MMASD với prediction clip đã lưu; phân tích đúng/sai toàn bộ, theo activity, joint và subject/session. Tách kết quả mô tả khỏi giả thuyết nhân quả. | Bảng/plot, effect size và khoảng tin cậy; báo cáo số clip thiếu prediction, class support nhỏ và confounder. |
| **P4 — Realistic corruption benchmark** | Dựa trên phân bố lỗi thật để chọn joint/vùng cơ thể, độ dài missing runs, tương quan thời gian và mức nhiễu. Áp corruption trên bản sao dữ liệu, giữ manifest và random seed; có clean/mild/medium/severe. | Generator và manifest tái lập được; corruption được kiểm tra đúng loại, vị trí và severity; dữ liệu raw không đổi. |
| **P5 — Baseline robustness** | Chạy checkpoint BiLSTM đã khóa trên clean và từng corruption/severity. Giữ tracking, preprocessing, windowing và aggregation đồng nhất; ghi rõ corruption đặt trước hay sau preprocessing. | Clip accuracy, macro-F1, per-class recall và `drop = metric_clean − metric_corrupted`; bảng/curve theo corruption và severity. |
| **P6 — Reliability-aware model** | Chỉ sau P2–P5 mới chọn phép thử có bằng chứng: thêm confidence, validity mask, hoặc pooling/weighting theo quality. So sánh công bằng với baseline và ablation. | Kết quả clean **và** corrupted, cùng split/protocol; báo cáo trường hợp cải thiện, suy giảm và giới hạn. |

### Những kiểm tra quan trọng trong P0–P2

- **MMASD:** định dạng JSON, số người/frame, selected index, target-track missing, raw confidence, `(0,0)`, mask trước điền thiếu, loại điền thiếu, tỷ lệ scale fallback. Đo trên chuỗi gốc trước resampling; sau đó đối chiếu những frame/window model thực sự dùng.
- **DREAM:** dùng độ dài mảng numeric thực, kiểm tra `null`, confidence, task metadata và frame rate. Không suy `end-start` là số frame khi metadata bất thường; không dựng trial correctness từ task metadata.
- **DRYAD:** dùng workbook skeleton theo frame để đo temporal/geometry; `Final dataset.xlsx` là đặc trưng tổng hợp nên không đủ thay cho chuỗi gốc. Gắn cờ giá trị `100` với ngữ nghĩa chưa rõ, không âm thầm sửa; tách original/augmentation và nhóm duplicate.
- **Tracking:** selected raw person index thay đổi chỉ là indicator. `possible_identity_switch_count` phải được đặt tên và định nghĩa như heuristic; `verified_identity_switch_count` chỉ có sau review danh tính có bằng chứng.
- **Representation:** thiếu chiều sâu ở OpenPose 2D là giới hạn biểu diễn; tách khỏi lỗi pose estimation, tracking và lỗi do preprocessing.

## 6. Hợp đồng đầu ra cho quality audit

Một dòng trong `skeleton_quality.csv` ứng với **một sequence/clip của một dataset và một processing variant**. Khóa tối thiểu: `dataset_id`, `clip_or_segment_id`, `processing_variant`, `source_version`; MMASD giữ thêm `clip_id`, `subject_id`, `session_id`, `activity`. Các cột ứng viên:

| Nhóm | Cột ứng viên | Cách hiểu |
|---|---|---|
| Provenance | `source_path`, `source_hash`, `config_hash`, `tracker_mode`, `preprocessing_version`, `sequence_mode` | Cho phép tái tạo và join đúng với checkpoint/prediction. |
| Quy mô | `num_raw_frames`, `num_model_frames`, `num_people_mean`, `valid_frame_ratio` | Phân biệt frame gốc, frame được chọn và frame tạo bởi resampling. |
| Missing/confidence | `missing_joint_ratio`, `max_continuous_missing`, `mean_confidence`, `median_confidence`, `low_confidence_ratio` | Tính trên raw selected skeleton; khai báo rõ confidence threshold và mẫu số. |
| Chuyển động/hình học | `jitter_score`, `bone_length_variation`, `joint_outlier_count` | Dùng đơn vị hoặc scale normalization rõ ràng; tránh coi chuyển động thật là jitter. |
| Tracking/preprocessing | `body_center_jump_count`, `possible_identity_switch_count`, `target_missing_frame_ratio`, `interpolated_joint_ratio`, `scale_fallback_ratio` | Indicator tự động, không phải verified error. |
| Trạng thái | `measurement_status`, `exclusion_reason`, `qc_notes` | Lưu trường hợp không có metric, malformed hoặc bị quarantine; không thay NA bằng 0. |

Lưu thêm bảng dài `skeleton_quality_joint.csv` (`clip_or_segment_id`, `joint_id`, missing/confidence/jitter theo joint) và, khi cần, bảng sự kiện theo frame để kiểm tra lại các spike. Không gộp các chỉ số thành một scalar “good/bad” ở bước đầu. Mỗi metric có định nghĩa, đơn vị, cách xử lý `null`, padding, zero và interpolation trong data dictionary.

## 7. Thực nghiệm đầu tiên: clip đúng và sai có quality khác nhau không?

1. **Khóa nguồn:** chọn checkpoint/config và prediction artifact làm phân tích chính trước khi xem kết quả QC; ghi split, danh sách clip và hash. Ưu tiên một model đơn đã có prediction cho 283 validation clip; dùng S1 275 clip như đối chiếu lịch sử với quy tắc loại 8 clip ngắn được công bố rõ. Không trộn prediction của nhiều experiment vào một nhóm.
2. **Join:** `clip_id` + split/variant; kiểm tra một dòng mỗi clip, `true_id` khớp manifest, `correct = (true_id == predicted_id)`, và báo cáo clip thiếu prediction. Không dùng hậu tố `_y/_i/_n` làm nhãn correctness, engagement hay quality.
3. **Metric chính:** missing-joint ratio, valid-frame ratio, confidence trung bình/median, max missing run, jitter, bone-length variation, center jumps và scale fallback; thêm các metric theo joint có giả thuyết vận động phù hợp (cổ tay cho drumming/clap, chân cho squat).
4. **Phân tích:** bảng mô tả đúng/sai; violin/ECDF hoặc boxplot theo metric, scatter với độ dài clip, per-class plot và heatmap theo joint. Báo cáo chênh lệch và khoảng tin cậy bằng bootstrap theo subject/session; kiểm tra nhạy với class, số frame và số người. Nếu dùng kiểm định, ưu tiên permutation hoặc regression có covariate đã khai báo; hiệu chỉnh khi kiểm tra nhiều joint/metric. Không diễn giải `p` hay correlation thành nguyên nhân.
5. **Diễn giải:** nêu metric nào đo trực tiếp từ dữ liệu, metric nào là proxy, giới hạn do thiếu RGB/identity labels. Xác định các failure pattern đáng ưu tiên cho P4, kể cả khi giả thuyết ban đầu không được hỗ trợ.

## 8. Quy tắc bằng chứng và giới hạn kết luận

- Confidence thấp là một tín hiệu reliability, không phải chứng cứ joint sai. Heuristic có thể gọi là *suspicious*, *low-confidence*, *temporally unstable* hoặc *geometrically inconsistent*; chỉ gọi “verified error” khi có chuẩn đối chiếu phù hợp.
- Không giải mã `_y`, `_i`, `_n` thành Yes/Incomplete/No khi chưa tìm được codebook đáng tin. [Báo cáo thực nghiệm](../reports/BAO_CAO_THUC_NGHIEM_BILSTM.md) có đoạn diễn giải hậu tố theo nghĩa này; kế hoạch **không kế thừa diễn giải đó** làm ground truth.
- Không tự chọn threshold “good/bad”. Nếu cần ngưỡng để tạo indicator hoặc severity, ghi rõ là heuristic, được hiệu chỉnh từ phân bố nào, và kiểm tra sensitivity.
- Không suy tỷ lệ nhận nhầm trẻ từ selected-index changes, model errors, kích thước skeleton hay ví dụ riêng lẻ. Review hiện có chưa có nhãn người được xác minh.
- Với mọi so sánh model/robustness, công bố split, checkpoint, preprocessing, corruption point, seed, aggregation và mẫu số. Không chọn cấu hình theo validation rồi gọi cùng validation là đánh giá độc lập.

## 9. Công việc tiếp theo ngay sau kế hoạch này

Thực hiện **P0 — audit repo và khóa protocol**. Bắt đầu bằng một báo cáo ngắn có bảy phần: (A) pipeline map theo file/hàm, (B) quality information đang có, (C) thông tin mất ở từng bước, (D) QC metric tính được không cần RGB, (E) metric chưa thể xác minh, (F) schema output, (G) thí nghiệm đúng/sai đầu tiên. Trước khi viết loader/metric mới, đối chiếu `improved_v1` với config/checkpoint thực tế và chọn prediction artifact chính. Không refactor hoặc huấn luyện model trong giai đoạn này.
