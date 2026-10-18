# Kế hoạch nghiên cứu: Đánh giá, phát hiện và hiệu chỉnh lỗi skeleton trẻ em

**Dự án:** NCKH Trẻ tự kỷ – Skeleton-based Autism Action Recognition  
**Ngày lập:** 01/10/2026  
**Cập nhật định hướng:** 14/10/2026, theo mục tiêu người dùng chốt.
**Trạng thái:** Kế hoạch nghiên cứu; chưa thay mô hình hoặc pipeline huấn luyện.

## 1. Mục tiêu và phạm vi

**Tên đề tài tạm thời:** Nghiên cứu phương pháp đánh giá và hiệu chỉnh chất lượng dữ liệu khung xương của trẻ từ video phục vụ phân tích hành vi.

**Mục tiêu đã chốt:** Nghiên cứu và xây dựng một pipeline tự động đánh giá, phát hiện và hiệu chỉnh các lỗi trong dữ liệu khung xương của trẻ em từ video, với trọng tâm kiểm chứng trên dữ liệu trẻ ASD, nhằm nâng cao độ tin cậy của dữ liệu chuyển động phục vụ phân tích hành vi.

Bốn mục tiêu cụ thể:

1. Khảo sát và chuẩn hóa dữ liệu skeleton từ các phiên hoạt động/trị liệu của trẻ, đặc biệt là dữ liệu ASD.
2. Xây dựng phương pháp đánh giá chất lượng và phát hiện missing joint, low-confidence, jitter/temporal outlier, bất thường hình học và mất tracking; phân biệt chỉ báo với lỗi có nhãn xác minh.
3. Đề xuất pipeline hiệu chỉnh bằng interpolation, temporal filtering, outlier correction và ràng buộc hình học/cơ thể khi biểu diễn dữ liệu cho phép.
4. Đánh giá trước/sau bằng metric định lượng và trực quan, phân tích theo activity, joint và điều kiện thu thập có bằng chứng.

ASD là miền ứng dụng và dữ liệu kiểm chứng chính; chưa giả định thuật toán phải đặc thù ASD. Chuyển động nhanh, nhiều người, tư thế đa dạng và occlusion là các yếu tố cần khảo sát, không mặc nhiên coi là đặc điểm riêng của ASD hay nguyên nhân lỗi đã được xác minh. Web/viewer là công cụ review và kiểm chứng. Nhận dạng hành vi là đánh giá downstream bổ sung; cải tiến kiến trúc phân loại không còn là mục tiêu chính.

**Phạm vi tuần đầu đã xác nhận:** người dùng chưa có video trẻ để nghiên cứu; dùng skeleton MMASD có sẵn. Chưa kiểm chứng bước trích xuất pose từ video mới. Khi có video phù hợp, bổ sung extractor và đánh giá đồng bộ RGB–skeleton. Bộ quy chuẩn/schema là nền tảng cho phương pháp, không thay thế đóng góp phát hiện/hiệu chỉnh và đánh giá.

Chuỗi công việc ưu tiên:

```text
VIDEO / SKELETON THÔ → CHUẨN HÓA VÀ GIỮ NGUỒN
                    → PHÂN TÍCH CHẤT LƯỢNG → PHÁT HIỆN LỖI
                    → HIỆU CHỈNH CÓ ĐIỀU KIỆN
                    → ĐÁNH GIÁ ĐỊNH LƯỢNG + TRỰC QUAN
```

Không gộp nhãn của MMASD, DREAM và DRYAD thành một bài toán phân loại. Không xem skeleton, confidence hay chỉ số heuristic là ground truth về tư thế hoặc danh tính người.

## 2. Bối cảnh dữ liệu và bằng chứng hiện có

| Dataset | Biểu diễn và nhãn | Vai trò trong hướng nghiên cứu |
|---|---|---|
| **MMASD** | OpenPose BODY_25 2D, nhiều người mỗi frame; 10 lớp hoạt động có OpenPose clip khả dụng trong pipeline hiện tại. | Case study chính: khảo sát chất lượng, phát hiện/hiệu chỉnh có điều kiện và đánh giá phục hồi/bảo toàn chuyển động; action recognition là phép kiểm chứng bổ sung. |
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

**Câu hỏi trung tâm:** Có thể tự động phát hiện và hiệu chỉnh các lỗi trong chuỗi skeleton của trẻ để cải thiện tính ổn định và độ tin cậy của dữ liệu chuyển động hay không?

1. **RQ1 — Khảo sát lỗi/chỉ báo:** Những dạng missing, confidence thấp, temporal instability, bất thường hình học và tracking ambiguity nào xuất hiện trên MMASD, với tần suất và độ dài bao nhiêu?
2. **RQ2 — Phụ thuộc điều kiện:** Phân bố các lỗi/chỉ báo khác nhau thế nào theo activity, joint, subject/session và số người? Chỉ phân tích occlusion/bối cảnh camera khi có RGB hoặc annotation/metadata hỗ trợ; thiếu joint không tự chứng minh occlusion.
3. **RQ3 — Hiệu quả phương pháp:** So với dữ liệu raw và các baseline, pipeline có cải thiện khả năng phục hồi trên lỗi được tạo có kiểm soát và các chỉ số chất lượng mà vẫn bảo toàn chuyển động không? Mức cải thiện và uncertainty là bao nhiêu?

Phép kiểm chứng bổ sung: input trước/sau hiệu chỉnh ảnh hưởng thế nào tới action recognition theo cùng protocol? Không yêu cầu kết quả phải dương; trường hợp thất bại là kết quả cần báo cáo.

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
| **P0 — Khảo sát và khóa protocol** | Audit parser/tracking/preprocessing, dữ liệu nguồn và literature về chất lượng/hiệu chỉnh skeleton; chọn pilot đa activity/subject/session. Tách nhóm hiệu chỉnh tham số và nhóm đánh giá theo subject/session, xác minh định danh nhóm. | Pipeline map, bảng phương pháp liên quan, manifest và protocol; checkpoint chỉ cần khóa nếu làm downstream. |
| **P1 — Chuẩn hóa có truy vết** | Schema sequence/frame/joint; giữ topology/units/confidence, raw/selected/processed, frame gốc và repair masks. Triển khai MMASD trước; DREAM/DRYAD là mở rộng. | Schema/data dictionary v0.1, adapter và export đọc lại được; source/config hash và quarantine report. |
| **P2 — Đánh giá và phát hiện** | Metric liên tục và detector theo từng loại; hiệu chỉnh threshold trên nhóm tuning, kiểm tra theo activity/joint/nhóm. Cảnh báo hình học 2D là proxy, không tự ép chiều dài xương cố định. | Quality tables, events với frame/joint/reason, detection config; đánh giá detector trên corruption có nhãn vị trí hoặc annotation có bằng chứng. |
| **P3 — Pipeline hiệu chỉnh** | Baseline nội suy ngắn; thêm một temporal filter; thử outlier correction chỉ khi tín hiệu phối hợp hỗ trợ. Ràng buộc hình học chỉ dùng khi đã xác minh representation và scale, không tự sửa tracking ambiguity. | Các variant raw/interpolation/filter/conditional correction; mask và repair log; không nối qua gap dài, track boundary hay imputed data giả làm quan sát. |
| **P4 — Benchmark kiểm soát** | Tạo missing runs và temporal spikes trên quan sát hợp lệ, nhiều mức/seed; giữ bản tham chiếu. Dựa vào audit để chọn tham số nhưng tách tuning/evaluation. | Corruption manifest có clip/joint/frame/severity/seed và vị trí nhãn; dữ liệu tham chiếu vẫn là output estimator, không phải ground truth thực. |
| **P5 — Đánh giá trước/sau** | So detection, recovery, unresolved coverage, jitter proxy và bảo toàn chuyển động; ablation trên cùng mẫu. Phân tích activity/joint/session/số người và review cả thành công/thất bại. | Bảng paired differences/effect size/uncertainty; detection precision/recall/F1 khi có nhãn phù hợp; reconstruction error và motion preservation. Không lấy smoothing làm chứng minh pose chính xác. |
| **P6 — Kiểm chứng bổ sung** | Downstream trên checkpoint có provenance; mở rộng dataset/hệ tọa độ đã xác minh; khi có video, bổ sung extraction và RGB annotation để xác minh pose/tracking/occlusion. | Kết quả có giới hạn phạm vi rõ; action recognition, extractor video và model reliability-aware là các nhánh mở rộng, không chặn P1–P5. |

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

## 7. Thực nghiệm cốt lõi: phát hiện và hiệu chỉnh có cải thiện dữ liệu không?

1. **Khóa mẫu:** pilot 20–50 clip MMASD có phân tầng activity/subject/session và độ dài; công bố coverage. Chọn 3–5 clip riêng phục vụ demo. Tách tuning/evaluation theo nhóm trước khi thử tham số; không dùng frame của cùng clip ở hai nhóm. Nếu nhóm quá ít, ghi kết quả là thăm dò.
2. **Khảo sát dữ liệu thật:** thống kê missing/confidence/gaps, temporal/geometric indicators và số người. Review các cảnh báo lẫn các đoạn không bị cảnh báo để kiểm tra false positives. Skeleton-only review chưa xác minh pose hoặc danh tính.
3. **Corruption có nhãn vị trí:** che joint theo đoạn và chèn temporal spikes vào bản sao; giữ mask/seed và reference. Không coi các vị trí còn lại là clean ground truth; metric detection chỉ đánh giá khả năng tìm lỗi chèn có kiểm soát trên tập ứng viên đã khai báo. Những lỗi thật nền chưa có nhãn làm giới hạn diễn giải precision.
4. **So sánh phương pháp:** B0 raw/no repair, B1 short-gap linear interpolation, B2 B1 + một temporal filter, B3 conditional outlier correction + B2. Chạy cùng corruption/seed/sample; bỏ từng thành phần để xác định nguồn thay đổi. Hình học 2D dùng để hỗ trợ cảnh báo; hard constraints trên cơ thể 3D là nhánh sau khi dữ liệu đủ căn cứ.
5. **Đo detection và recovery:** precision/recall/F1 với nhãn phù hợp, lỗi tọa độ chuẩn hóa theo scale tham chiếu cố định khi hợp lệ, tỷ lệ sửa được/còn thiếu và sai số theo gap length/joint/activity. Transform trong pipeline chỉ được fit từ dữ liệu khả dụng sau corruption; không dùng điểm bị che để suy root/scale/filter.
6. **Đo bảo toàn chuyển động:** displacement, biên độ/quỹ đạo và thời điểm cực trị trên đoạn tham chiếu; báo cáo độ trễ/biến dạng và số điểm tốt bị sửa. Filter chạy riêng trên các đoạn hợp lệ liên tục, không bắc qua track boundary hoặc missing gap chưa được xử lý. Phương pháp offline được dùng tương lai nhưng không được mô tả là realtime/causal.
7. **Đánh giá thống kê và trực quan:** paired differences theo clip và khoảng tin cậy bootstrap theo subject/session khi số nhóm cho phép; không coi hàng nghìn frame là mẫu độc lập. Báo cáo effect size, failure cases và uncertainty; chỉnh nhiều joint/metric phải xử lý multiple comparisons nếu công bố kiểm định. Không mặc định kết luận có ý nghĩa thống kê.
8. **Downstream bổ sung:** khi đủ provenance, so action recognition trên cùng clip/split/tracking/preprocessing/window/aggregation; xử lý mới không bị preprocessing hai lần. Dùng validation đúng tên, báo clip thiếu prediction, và không dùng hậu tố `_y/_i/_n` làm correctness. Phép phân tích quality–prediction cũ vẫn hữu ích như bằng chứng bổ sung.

Tài liệu phương pháp để đối chiếu: [Savitzky–Golay trong SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) và [1€ Filter của nhóm tác giả](https://gery.casiez.net/1euro/). Chọn một filter làm baseline tuần đầu sau khi kiểm tra sampling và điều kiện missing; không mặc định filter nào tốt nhất trên MMASD.

Đóng góp cần chứng minh là lựa chọn/tổ chức phát hiện–hiệu chỉnh có điều kiện phù hợp dữ liệu khảo sát và bằng chứng so sánh, không chỉ việc ghép các thuật toán đã có. Đặc thù dữ liệu trẻ/ASD phải được phân tích từ dữ liệu; chưa có nhóm so sánh thì không kết luận lỗi phổ biến hơn người lớn hoặc TD.

## 8. Quy tắc bằng chứng và giới hạn kết luận

- Confidence thấp là một tín hiệu reliability, không phải chứng cứ joint sai. Heuristic có thể gọi là *suspicious*, *low-confidence*, *temporally unstable* hoặc *geometrically inconsistent*; chỉ gọi “verified error” khi có chuẩn đối chiếu phù hợp.
- Không giải mã `_y`, `_i`, `_n` thành Yes/Incomplete/No khi chưa tìm được codebook đáng tin. [Báo cáo thực nghiệm](../reports/BAO_CAO_THUC_NGHIEM_BILSTM.md) có đoạn diễn giải hậu tố theo nghĩa này; kế hoạch **không kế thừa diễn giải đó** làm ground truth.
- Không tự chọn threshold “good/bad”. Nếu cần ngưỡng để tạo indicator hoặc severity, ghi rõ là heuristic, được hiệu chỉnh từ phân bố nào, và kiểm tra sensitivity.
- Không suy tỷ lệ nhận nhầm trẻ từ selected-index changes, model errors, kích thước skeleton hay ví dụ riêng lẻ. Review hiện có chưa có nhãn người được xác minh.
- Với mọi so sánh model/robustness, công bố split, checkpoint, preprocessing, corruption point, seed, aggregation và mẫu số. Không chọn cấu hình theo validation rồi gọi cùng validation là đánh giá độc lập.

## 9. Công việc tiếp theo ngay sau kế hoạch này

Thực hiện **P0–P1 trên skeleton MMASD**, theo [lịch 7 ngày](KE_HOACH_7_NGAY_SAN_PHAM_CHUAN_HOA_SKELETON.md). Đầu ra đầu tiên: `protocol.md`, manifest pilot/tuning/evaluation, schema/data dictionary, pipeline map và bảng khảo sát phương pháp. Sau đó triển khai quality/events → short-gap interpolation → một filter → conditional outlier correction → benchmark và báo cáo. Viewer hiển thị chính artifact pipeline xuất ra.

Không yêu cầu video hoặc checkpoint để bắt đầu công việc cốt lõi này. Bất nhất checkpoint chỉ cần giải quyết trước phép kiểm chứng nhận dạng; việc chưa có RGB/3D metric đã xác minh phải được ghi trong giới hạn. Chưa huấn luyện mô hình nhận dạng mới trong tuần đầu.
