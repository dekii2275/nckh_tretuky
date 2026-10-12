# Kế hoạch 6–7 ngày: kiểm toán skeleton MMASD và demo nghiên cứu

**Trạng thái:** kế hoạch thực hiện; các ô trong bảng chưa đánh dấu là việc chưa hoàn thành.  
**Thời gian:** 6 ngày làm phần cốt lõi, ngày 7 dành cho kiểm tra và hoàn thiện. Tính D1 từ ngày bắt đầu thực tế.  
**Phạm vi chính:** MMASD OpenPose BODY_25 2D; phân tích ở cấp clip, frame và joint.  
**Hướng nghiên cứu:** [Skeleton Reliability và Robust Action Recognition](KE_HOACH_NGHIEN_CUU_SKELETON_RELIABILITY.md).

## 1. Kết quả cần có cuối đợt

1. Một demo mở trong trình duyệt: chọn một số clip MMASD, phát/tua skeleton 2D, xem confidence/missingness và các frame **đáng kiểm tra**, đối chiếu raw với biến thể xử lý trên cùng frame và hệ tọa độ.
2. Một bộ kết quả có thể chạy lại: định nghĩa metric, bảng chất lượng theo clip/joint, phép join với prediction và phân tích quality so với lỗi nhận dạng hoạt động.
3. Một thí nghiệm robustness nhỏ: 1–2 loại corruption có seed và mức độ cố định; đánh giá checkpoint đã khóa trên clean và corrupted bằng cùng protocol.
4. Báo cáo ngắn ghi số mẫu, dữ liệu thiếu, giới hạn kết luận và hình cho buổi trình bày 3–5 phút.

**Điều kiện hoàn thành:** người khác có thể xem demo và chạy lại bảng số liệu từ config, manifest và lệnh được ghi trong README. Không dùng hình ảnh demo làm bằng chứng định lượng nếu chưa có phép đo tương ứng.

## 2. Quy tắc phạm vi và bằng chứng

- Dùng lại [parser/tracking/preprocessing hiện có](../../BiLSTM/src/data/) trước khi viết pipeline khác. Giữ dữ liệu gốc bất biến; đầu ra mới đặt dưới `BiLSTM/artifacts/reliability_week1/` hoặc thư mục artifact tương đương.
- D1 phải chọn **một** checkpoint, config, split và tập prediction làm phân tích chính. Đối chiếu bất nhất: [báo cáo hiện tại](../reports/BAO_CAO_THUC_NGHIEM_BILSTM.md) mô tả `improved_v1` bằng Tracker V2, còn [config](../../BiLSTM/configs/improved_v1.yaml) ghi `romp_guided`. Ghi rõ nguồn nào thực sự tạo checkpoint; nếu chưa xác nhận được, chọn experiment khác có provenance rõ.
- Giữ các tầng `raw candidates → selected person → validity mask → processed/corrected → model input` riêng. Mỗi sự kiện phải truy về `clip_id`, frame gốc, joint, tracker/config và phiên bản xử lý. FPS chỉ ghi khi có nguồn xác nhận; nếu chưa có, hiển thị số frame thay cho thời gian tính từ FPS giả định.
- Confidence thấp, chuyển động đột ngột, hình học bất thường và thay đổi chỉ số người là **chỉ báo**, không phải lỗi đã xác minh. Không tạo một `quality score` tổng hợp hoặc phần trăm “good/bad” khi chưa có ngưỡng được hiệu chỉnh. [Audit tracking hiện có](../../BiLSTM/artifacts/diagnostics/tracking_audit_summary.md) chưa có clip được người đánh giá.
- Biến thể “corrected” tuần này chỉ là **baseline nội suy có điều kiện** trên gap ngắn, được đánh dấu rõ khớp nào đã ước lượng. Không tự gán confidence cao cho khớp nội suy. Không suy cải thiện độ chính xác chỉ từ missing ratio hoặc jitter giảm.
- Bản MMASD cục bộ không có video RGB gốc. Demo tuần này dùng skeleton 2D; video overlay, ROMP 3D, upload video, annotation UI đầy đủ và mở rộng DREAM/DRYAD nằm ngoài phạm vi 6 ngày cốt lõi.
- Tập đang gọi là **validation**, không gọi là test độc lập. [Phân tích S1 sẵn có](../../BiLSTM/artifacts/diagnostics/error_factor_analysis.md) có prediction cho 275 clip và thiếu 8 clip ngắn; số lượng cuối cùng phải đối chiếu lại với artifact đã khóa.

## 3. Bảng checklist thực hiện

Đổi `☐` thành `☑` khi **đầu ra/tiêu chí kiểm tra** ở cột cuối đã có. Lưu đường dẫn artifact hoặc commit cạnh dòng tương ứng nếu cần bàn giao.

| Ngày | Trạng thái | Công việc | Đầu ra / tiêu chí kiểm tra |
|---|:---:|---|---|
| **D1 — Khóa protocol** | ☐ | Chọn checkpoint, config, split, prediction artifact và ghi hash/phiên bản. | `protocol.md`; một bảng provenance không còn nhầm Tracker V2 với `romp_guided`. |
| D1 | ☐ | Kiểm tra manifest và join `clip_id`; đếm clip có/không có prediction, lớp và subject/session. | `join_audit.csv` hoặc bảng tương đương; ghi rõ mẫu số phân tích. |
| D1 | ☐ | Phát được một clip OpenPose BODY_25 trong trình duyệt. | Có play, pause, seek và frame index; lưu chỉ số raw candidate được chọn ở từng frame. |
| **D2 — Hợp đồng dữ liệu và viewer** | ☐ | Chuẩn hóa output của vài clip mẫu, giữ raw candidate, selected person, confidence, validity mask, frame gốc và topology. | Tài liệu schema ngắn; đọc lại được clip mẫu mà không đọc format MMASD trong frontend. |
| D2 | ☐ | Thêm danh sách clip nhỏ và viewer 2D có khung hình cố định, tên joint và confidence. | Chọn được 3–5 clip đại diện; play/seek đúng frame; khớp thiếu không vẽ như tọa độ `(0,0)`. |
| D2 | ☐ | Định nghĩa cách so raw/processed ở cùng hệ tọa độ, cùng người được chọn và cùng frame. | Một ví dụ kiểm tra được bằng mắt; không so pixel gốc với tọa độ chuẩn hóa như thể cùng đơn vị. |
| **D3 — Quality audit** | ☐ | Tính missing-joint ratio, confidence và độ dài gap trên selected raw, trước khi nội suy/resampling. | `quality_per_clip.csv`, `quality_per_joint.csv`; có đơn vị, mẫu số và quy tắc NA. |
| D3 | ☐ | Thêm 1–2 indicator chuyển động/hình học, tính ở đơn vị hoặc scale đã khai báo. | `events.csv` có frame/joint và lý do; không gọi indicator là verified error. |
| D3 | ☐ | Làm baseline nội suy tuyến tính chỉ cho gap được chặn hai đầu và ngắn; lưu mask nội suy. | Raw giữ nguyên; viewer tô/đánh dấu vị trí được ước lượng; gap dài để thiếu. |
| **D4 — Quality và lỗi nhận dạng** | ☐ | Join quality với prediction đã khóa bằng `clip_id` và variant; kiểm tra nhãn thật khớp manifest. | Một hàng/clip; bảng số clip join được, thiếu và bị loại. |
| D4 | ☐ | So sánh clip đúng/sai theo missingness, confidence, activity, độ dài clip và subject/session. | Bảng chênh lệch cùng 2–3 hình; nêu yếu tố gây nhiễu và không diễn giải là quan hệ nhân quả. |
| D4 | ☐ | Chọn 20–50 sự kiện/frame **có chủ đích** để xem lý do bị gắn cờ và ghi nhận xét thủ công. | Manifest lựa chọn và ghi chú; không gọi nhận xét trên skeleton là xác minh pose/danh tính khi thiếu RGB. |
| **D5 — Corruption benchmark** | ☐ | Tạo tối đa hai loại corruption: mất khớp theo đoạn và nhiễu tọa độ. | Generator nhận seed/config; không ghi đè raw; có clean, mild, medium, severe. |
| D5 | ☐ | Chèn corruption vào selected raw sau tracking, trước điền thiếu/chuẩn hóa, rồi kiểm tra vài clip. | `corruption_manifest.csv` ghi clip, joint, frame, severity, seed; tracking giữ cố định và lỗi mô phỏng đúng cấu hình. |
| **D6 — Đánh giá và kết luận** | ☐ | Chạy checkpoint cố định trên clean và từng corruption/severity với cùng tracking, preprocessing, windowing, aggregation. | `robustness_results.csv`: clip accuracy, macro-F1, per-class recall, số clip và chênh lệch so với clean. |
| D6 | ☐ | Kiểm tra baseline nội suy bằng che thử một số khớp vốn quan sát được; nếu đủ thời gian, đo ảnh hưởng tới prediction. | Sai số khôi phục theo độ dài gap so với pose quan sát được; đây chưa phải ground truth về pose thực. |
| D6 | ☐ | Viết báo cáo ngắn và README chạy lại. | Câu hỏi, protocol, bảng số liệu, giới hạn, lệnh chạy và đường dẫn artifact đầy đủ. |
| **D7 — Dự phòng và trình bày** | ☐ | Sửa lỗi demo, kiểm tra clip hỏng/thiếu, hoàn thiện timeline và 2–3 hình quan trọng. | Demo chạy trọn flow 3–5 phút trên máy trình bày; không có số liệu giả hoặc biểu đồ placeholder. |
| D7 | ☐ | Rà soát mọi tuyên bố và bàn giao. | Phân biệt indicator với nhãn xác minh, validation với test, raw với processed; ghi việc chưa xong và hướng tiếp theo. |

## 4. Thứ tự ưu tiên khi chỉ có 6 ngày

1. **Phải xong:** D1 protocol; D2 viewer 2D; D3 quality tables; D4 phân tích đúng/sai; D5–D6 ít nhất một corruption và một đường đánh giá clean/corrupted; README tái lập.
2. **Giảm trước nếu thiếu thời gian:** số clip được chuẩn bị cho viewer, số indicator, số loại corruption, độ bóng bẩy của dashboard. Phân tích định lượng vẫn dùng toàn bộ tập validation **có prediction hợp lệ** của checkpoint đã khóa; 20–50 sự kiện/frame chỉ là mẫu xem xét thủ công.
3. **Chuyển sang giai đoạn sau:** 3D, video overlay/upload, đổi sang MediaPipe, nhiều bộ dữ liệu trên cùng dashboard, annotation UI phức tạp, quality score tổng hợp, huấn luyện mô hình mới.

Nếu provenance checkpoint hoặc join prediction chưa rõ vào cuối D1, tiếp tục đo chất lượng và làm viewer nhưng **không công bố** so sánh recognition/robustness như kết quả đã khóa protocol. Nếu không chạy kịp inference ở D6, báo cáo phần quality và đánh dấu robustness là chưa hoàn thành, không điền số ước đoán.

## 5. Mẫu câu kết luận phù hợp với mức bằng chứng

> Trên tập validation và protocol đã nêu, các clip dự đoán sai có một số chỉ báo chất lượng skeleton khác với clip đúng. Các chỉ báo này chưa xác minh vị trí khớp hay danh tính người. Thử nghiệm corruption cho thấy mức thay đổi hiệu năng nhận dạng trong các điều kiện mô phỏng đã định nghĩa; kết quả chưa chứng minh thuật toán nội suy sửa đúng pose thực tế.
