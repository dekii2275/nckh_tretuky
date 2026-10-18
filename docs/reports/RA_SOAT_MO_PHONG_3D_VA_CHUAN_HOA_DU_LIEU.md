# Rà soát phần hiển thị 3D và hướng chuẩn hóa dữ liệu

Ngày rà soát: 14/10/2026. Phạm vi chính: viewer MMASD trong `BiLSTM/scripts/export_tracking_review_3d.py`, dữ liệu nguồn của 120 clip trong review manifest, HTML đã xuất và loader BiLSTM hiện tại. DREAM/DRYAD được đối chiếu ở mức kiến trúc và cấu hình; chưa audit lại toàn bộ hai dataset này. Rà soát bằng đọc mã và dữ liệu, chưa kiểm tra bằng trình duyệt.

## Kết luận

Phần đang gọi là mô phỏng 3D là **phát lại skeleton ROMP đã có**, bằng phép chiếu lên canvas. Script không chạy ROMP để tái dựng pose, không tạo mesh/avatar SMPL và không mô phỏng vật lý. Viewer hữu ích để xem chuyển động nhưng cần sửa liên kết người, xác minh hệ tọa độ và dùng cùng pipeline với model trước khi làm công cụ duyệt dữ liệu chuẩn hóa.

Ưu tiên: **khóa nguồn và schema → xác minh thời gian/người/hệ tọa độ → xử lý missing và normalization → viewer so sánh → tạo dataset phiên bản mới**. Chuẩn hóa số ngay bây giờ có thể che mất lỗi ghép người và lỗi đơn vị.

## 1. Luồng tại thời điểm rà soát ban đầu

1. Đọc `BiLSTM/artifacts/tracking_review/review_manifest.csv`; mặc định xuất 10 clip, `--limit 0` lấy toàn bộ danh sách.
2. Đọc OpenPose BODY_25 JSON và ROMP NPZ trường `coordinates` từ hai thư mục modality. Danh sách file được sắp xếp riêng.
3. Cắt cả hai về `min(len(2D), len(3D))`; không join bằng frame ID/timestamp.
4. Gọi trực tiếp `select_child_track` trên OpenPose, dùng `paper_repro.yaml`. Không gọi dispatcher `select_person_track` mà loader huấn luyện sử dụng.
5. Đóng gói raw candidates và `selected_indices` thành HTML. Tọa độ được làm tròn hai chữ số thập phân. ROMP person không hợp lệ bị loại khỏi danh sách; lỗi đọc NPZ được đổi thành frame rỗng, không lưu nguyên nhân.
6. Canvas 3D lấy tâm trung bình của tất cả khớp/người/frame, đảo Y, xoay yaw/pitch rồi chiếu phối cảnh. Camera, mặt sàn và kích thước lưới dùng hằng số. Đây là biến đổi hiển thị, không phải normalization cơ thể.
7. Cả canvas 2D và 3D dùng cùng `selected_indices` của OpenPose. Chưa có mapping OpenPose person → ROMP person.
8. Phát mỗi 100 ms tại 1x, tương đương 10 frame/s, không đọc FPS nguồn. Review lưu vào browser localStorage và tải CSV khi người dùng xuất; chưa áp dụng ngược vào loader.

Vị trí mã chính:

- [Đọc ROMP](../../BiLSTM/scripts/export_tracking_review_3d.py), hàm `load_romp_3d_clip`.
- Cùng file: `main`, `project3D`, `render3D`, `togglePlay`, `getReviewObject`.
- [Loader model](../../BiLSTM/src/data/dataset.py), `SkeletonClipDataset.__getitem__`.
- [Tracker dispatcher](../../BiLSTM/src/data/tracking.py), `select_person_track`.
- [ROMP-guided tracker](../../BiLSTM/src/data/romp_guided_tracking.py): dùng **ROMP 2D bounding boxes**, không dùng Z của ROMP 3D.
- [Tiền xử lý](../../BiLSTM/src/data/preprocessing.py), `preprocess_clip`.

## 2. Kiểm tra trực tiếp dữ liệu

Hai phép kiểm tra có phạm vi khác nhau, không suy rộng thành audit toàn bộ MMASD:

| Kiểm tra | Kết quả |
|---|---|
| Danh sách review | 120 clip; thư mục viewer có 120 trang clip và một index |
| File ID trong ba modality, trên 120 clip | Các danh sách ID OpenPose/ROMP 2D/ROMP 3D bằng nhau; không thấy thiếu ID ở giữa hoặc lệch số file |
| NPZ raw tại frame đầu, giữa, cuối mỗi clip | 360 frame; tất cả có shape `[P,24,3]`, không thấy NaN/Inf trong các mẫu này |
| So ROMP 2D với X/Y của ROMP 3D ở các mẫu | Trùng chính xác ở 360/360 frame |
| Số người OpenPose khác ROMP ở các mẫu | 171/360 frame |
| Đọc toàn bộ payload HTML của 120 clip | 27.374 frame |
| Số người 2D khác 3D trong HTML | 10.525/27.374 frame, khoảng 38,45% |
| OpenPose selected index vượt danh sách ROMP trong HTML | 2.087/27.374 frame, khoảng 7,62% |

Ví dụ `as_20613_D1_002_i`, frame zero-based 37: OpenPose có 6 người, ROMP có 4 người, tracker chọn OpenPose index 4. Viewer tìm ROMP index 4 nên không có người 3D nào nhận nhãn TARGET tại frame đó. Khi index nằm trong phạm vi, cũng chưa có bằng chứng đó là cùng người.

Phương pháp: đọc tên file trong ba thư mục của từng dòng review manifest, trích frame ID bằng hậu tố số; đọc trường `coordinates` với `np.load(..., allow_pickle=False)` ở ba vị trí; đọc JSON trong `<script id="dataPayload">` của từng HTML rồi so số người và selected index theo frame. Số liệu HTML phản ánh artifact đang có, không phải một lần regenerate mới.

**Cần xác minh hệ tọa độ:** các mẫu ROMP 3D giữ X/Y giống hệt ROMP 2D. Ví dụ frame 0 của clip trên, X khoảng 145–421, Y khoảng 15–341, Z khoảng 124–554. Chưa tìm thấy metadata trong các NPZ mẫu ngoài `coordinates` giải thích đơn vị Z, calibration hoặc phép biến đổi từ ROMP nguyên bản. Điều này chưa chứng minh dữ liệu sai, nhưng chưa đủ để coi XYZ là tọa độ Cartesian cùng đơn vị hoặc gán mét/cm. Nhãn chiều cao `px` trong viewer không xác minh được đơn vị 3D.

**Cần quản lý phiên bản topology:** README MMASD cục bộ và [tài liệu ROMP chính thức](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md) mô tả output 71 khớp, gồm 24 SMPL + 30 extra + 17 H36M. Trong khi đó, nguồn viewer được lấy mẫu ở đây có 24 khớp. Đây là hai biểu diễn cần ghi riêng; shape 24 không tự xác nhận tên/thứ tự khớp. Viewer đang hard-code graph SMPL_24. Phải tìm provenance của bản xuất 24 khớp trước khi khẳng định mapping đúng.

## 3. Những điểm ảnh hưởng trực tiếp tới chuẩn hóa

| Mức ưu tiên | Điểm cần xử lý | Hệ quả |
|---|---|---|
| P0 | Dùng OpenPose index cho ROMP | Tô nhầm người hoặc không có TARGET 3D; review không đủ căn cứ xác nhận child track |
| P0 | Thiếu đơn vị/hệ tọa độ và provenance bản 24 khớp | Khoảng cách, góc 3D, độ dài xương và scale có thể không mang ý nghĩa hình học mong đợi |
| P0 | Viewer dùng tracker riêng và config cố định | Không đảm bảo người reviewer thấy là người checkpoint khác thực sự dùng |
| P1 | Cắt theo độ dài và ghép bằng vị trí | Chưa gây mismatch tên frame ở 120 clip kiểm tra, nhưng sẽ che lỗi ở nguồn thiếu frame hoặc có offset |
| P1 | Lọc bỏ cả người nếu một giá trị không finite, bỏ index gốc | Mất khớp tốt còn lại và thay đổi chỉ số ứng viên; không truy vết được lý do thiếu |
| P1 | Chỉ hiển thị raw; ngưỡng vẽ 0,05 khác ngưỡng tracking 0,1 | Chưa thấy dữ liệu sau nội suy, normalization và sampling mà model nhận |
| P1 | Không có FPS nguồn; review chủ yếu cấp clip | Không đo vận tốc theo giây chính xác và khó chỉ định đoạn cần sửa |
| P1 | Review lưu localStorage, CSV 3D khác schema audit 2D | Chưa có đường nhập review có version để ảnh hưởng có kiểm soát tới dữ liệu |

Loader BiLSTM hiện nhận tọa độ **2D**: chọn người → tạo validity mask → điền thiếu → trừ MidHip → chia chiều rộng vai với fallback thân → smoothing tùy config → sampling/window → model. Confidence/mask không mặc nhiên là input model; mask chỉ trả ra ở chế độ debug. `romp_guided` hỗ trợ chọn OpenPose person bằng ROMP 2D, chưa phải học skeleton 3D.

Trong `paper_repro.yaml`, `max_gap: null` cho phép điền khoảng thiếu dài và kéo giá trị quan sát tới mép sequence. Vì vậy, cần phân biệt observed / interpolated / edge-filled / never-observed, thay vì coi tất cả tọa độ hữu hạn là dữ liệu đo được. Việc đổi policy này sẽ thay đổi đầu vào model và cần dataset/config/checkpoint mới.

## 4. Thiết kế chuẩn hóa đề xuất

### 4.1. Tạo bản dữ liệu có truy vết

Giữ bốn lớp: `raw` → `selected` → `processed` → `model_input`. Viewer đọc từ cùng output trung gian với loader, tránh có pipeline xử lý riêng trong frontend.

Metadata tối thiểu:

- Nguồn: `dataset_id`, `clip_id`, `subject_id`, `session_id`, source path/hash, source version, config/code hash, schema version.
- Thời gian: source frame ID/path, timestamp, FPS và trạng thái đã xác minh/suy ra/chưa biết. Frame ID không tự chứng minh đồng bộ cảm biến nếu chưa biết offset/FPS.
- Hình học: topology ID, joint names/edges/mapping version, coordinate space, axis convention, unit per axis, calibration/transform provenance. Nếu chưa biết thì ghi `unknown`.
- Người: source candidate index cho từng modality, track ID theo thời gian, selected index riêng cho 2D và 3D, match score/status và vai trò đã review hoặc `unknown`.
- Chất lượng từng khớp: raw confidence nếu có, observed mask, imputed mask, missing reason, repair method. Confidence không có phải để null, không gán 1.
- Biến đổi: root/scale/rotation đã áp dụng, fallback và mapping từ model frame về source frame/time, gồm interpolation weights hoặc padding mask khi cần.

Không cần lưu tất cả dữ liệu nhiều lần nếu có thể tái tạo từ nguồn bất biến; cần bảo đảm từng lớp đọc lại được và có provenance.

### 4.2. Ghép frame và người trước khi sửa pose

- Join bằng source frame ID hoặc timestamp đã xác minh; giữ frame thiếu bằng mask, báo unmatched frames. Không âm thầm cắt về độ dài ngắn nhất.
- Xác minh ROMP 2D và 3D giữ cùng thứ tự candidate ở cùng frame; sau đó match OpenPose với ROMP 2D bằng joint mapping đã kiểm chứng, khoảng cách chuẩn hóa theo scale và kiểm tra bbox.
- Cho phép `unmatched` / `ambiguous`, không luôn ép chọn ứng viên gần nhất. Dùng liên tục theo thời gian và tracklet để giảm đổi danh tính; raw array index không phải track ID.
- Giữ candidate index gốc sau mọi thao tác lọc. Missingness nên quản lý từng khớp; file hỏng có reason/path và được quarantine khi cần.
- Người nhỏ nhất chỉ là heuristic; review có lựa chọn chưa xác định. Đặc biệt cần xem đoạn crossing, người ra/vào hình và đổi số ứng viên.

### 4.3. Chuẩn hóa hình học theo nguồn và tác vụ

Trước hết xác minh provenance X/Y/Z và mapping 24 khớp. Nếu chỉ có tọa độ phục vụ hiển thị với Z chưa rõ, giữ nhánh này cho visualization/audit; cần nguồn ROMP 3D đã xác minh hoặc phép chuyển đổi có tài liệu trước khi dùng để tính hình học metric.

Với 3D Cartesian đã xác minh, đề xuất representation cho nhận dạng tư thế:

`normalized[t,j] = R[t] @ (position[t,j] - root[t]) / scale_clip`

- Root theo topology: pelvis/MidHip khi có. DREAM không có hông nên dùng shoulder center cho representation thân trên và ghi rõ khác biệt.
- `scale_clip` là scale cơ thể robust từ các frame hợp lệ (ví dụ median chiều dài thân); scale nhất quán theo clip hạn chế nhiễu từ scale đổi từng frame. So sánh với baseline hiện tại trước khi chốt.
- `R` có thể là identity trong bản đầu. Nếu chuẩn hóa hướng cơ thể, giữ cả hướng gốc và transform; không tự loại hướng nhìn/xoay khi đó là tín hiệu tác vụ.
- Giữ riêng root trajectory và scale gốc để không mất thông tin di chuyển hoặc kích thước cần cho tác vụ gait.
- Với 2D, chuẩn hóa trong mặt phẳng ảnh và đánh giá riêng; không xem shoulder width 2D là số đo cơ thể 3D.
- Chỉ nội suy khoảng thiếu ngắn, được kẹp bởi quan sát của cùng track. Chọn giới hạn bằng thời gian và kiểm tra trên train/validation; giữ mask cho khoảng dài và không tự nối qua identity switch.
- Chỉ fit statistical normalization trên train. Sampling 60 frame/window thực hiện sau audit và phải giữ liên kết thời gian nguồn.

### 4.4. Dùng viewer để kiểm tra kết quả xử lý

Ba chế độ tối thiểu: raw candidates, selected raw, processed/model input. Dùng source frame/time chung; có chọn khớp để xem tọa độ, confidence, validity, cách sửa và transform. Khi so raw với normalized, đưa về cùng hệ qua inverse transform hoặc ghi rõ hệ/đơn vị ở mỗi panel.

Chỉ tô TARGET 3D khi mapping đã xác lập; hiển thị unmatched/ambiguous. Camera auto-fit theo bounds hợp lệ, FPS nguồn có provenance, trục tọa độ và trạng thái đơn vị. Mặt sàn chỉ thể hiện mặt sàn đã xác minh hoặc ghi rõ là lưới tham chiếu.

Review lưu artifact có version và source/config hash; bổ sung frame interval, người đúng theo modality và trạng thái không chắc chắn. Việc review → override track / loại đoạn / quarantine cần là bước import riêng có báo cáo, không chỉ lưu trong browser.

## 5. Lộ trình và điều kiện nghiệm thu

| Thứ tự | Công việc | Đầu ra và điều kiện đạt |
|---|---|---|
| 1 | Audit nguồn, topology, đơn vị, thời gian toàn bộ MMASD | Manifest nguồn, histogram shapes, lỗi đọc, frame mismatch, unknown metadata; xác minh hoặc quarantine nguồn Z chưa rõ |
| 2 | Schema và association trên 120 clip review | Selected indices riêng cho từng modality; mọi match có trạng thái; không dùng index 2D trực tiếp cho 3D; lưu raw index gốc |
| 3 | Dùng cùng config/dispatcher/output với loader và sửa viewer | Viewer cho thấy raw/selected/processed, mask và source frame; đối chiếu bằng mắt các trường hợp có/không có người, switch, missing |
| 4 | Thử normalization trên tập nhỏ đa lớp/subject/session | Báo cáo missingness trước/sau, scale/root fallback, độ ổn định và biến đổi bone lengths trong cùng hệ đã xác minh; không sửa xương chỉ để làm đẹp |
| 5 | Xuất dataset version mới và đánh giá đối chứng | Giữ baseline/split cũ; so chất lượng, motion preservation và hiệu năng trên protocol cố định; mọi thay đổi có provenance |

Kiểm thử khi triển khai cần tập trung vào lỗi có ý nghĩa: frame bị thiếu ở giữa; số/thứ tự người đổi; candidate không match; joint NaN nhưng khớp khác vẫn tốt; shape 24/71; tọa độ hợp lệ ở origin; interpolation không vượt gap/track; model-frame mapping và inverse transform.

Nếu mở rộng sang DRYAD/DREAM, chỉ thống nhất schema/truy vết và graph thân trên khi cần so sánh. Giữ topology native, đơn vị và nhãn tác vụ riêng. DRYAD hiện ghi `normalization: none` trong preprocessing config; DREAM viewer giữ confidence/null và đọc `frame_rate`. Không lấy transformation hiển thị của MMASD làm chuẩn chung cho hai nguồn này.

## 6. Trạng thái sau lần rà soát ban đầu

Chỉ bổ sung báo cáo và liên kết chỉ mục. Chưa sửa viewer/loader/config, chưa regenerate HTML/dataset, chưa huấn luyện lại và chưa thay đổi dữ liệu gốc. Các số liệu về candidate mismatch là bằng chứng lỗi sử dụng index giữa hai modality, không phải tỷ lệ tracker chọn sai trẻ hay độ chính xác ROMP.

## 7. Cập nhật viewer theo yêu cầu kiểm tra mapping 3D

Sau lần rà soát ban đầu, người dùng cung cấp ảnh frame 25 của `as_20613_D1_002_i` và yêu cầu sửa hình 3D trông ngược, thể hiện cơ thể đầy đủ hơn. Đã sửa viewer và xuất lại danh sách 120 clip:

- Đối chiếu graph với thứ tự SMPL24 được ROMP công bố. Các cạnh hông–gối–cổ chân, vai–khuỷu–cổ tay và cột sống hiện khớp graph đó; joint 15 được sửa tên từ `Head` thành `Jaw`. Trong các nguồn 71 khớp, chỉ lấy core 24 khớp đầu, không nối thêm khớp bằng chỉ số tùy ý. Provenance bản export nguồn vẫn cần được bổ sung.
- Lỗi mặt phẳng cũ xuất phát từ `2 * centerY - 280`, khiến lưới nằm phía trên người. Renderer mới đặt mặt phẳng tham chiếu dưới bounds hợp lệ. Hướng pitch được sửa để camera phía trên thực sự nhìn xuống; mặc định nhìn chính diện với phép chiếu song song, giữ X/Y tương ứng ảnh. Không tự đổi dấu Z khi chưa xác minh convention nguồn.
- Thêm bề mặt thân, đầu, tay, chân, bàn tay và bàn chân dựng từ khớp bằng mesh hình học; có chế độ cơ thể, cơ thể + khớp, và xương gốc. Đây là mesh minh họa, không phải mesh SMPL nguyên bản phục hồi từ pose/shape. Mũi minh họa thể hiện hướng body được suy từ vai/thân, không phải phép đo gaze.
- Có chọn tất cả người, ứng viên được ghép hoặc từng ROMP index; camera theo pelvis trong chế độ tập trung và bounds được giữ cố định theo clip để không tự đổi scale theo mỗi frame. Viewer không sửa tọa độ nguồn để làm đẹp pose.
- Join theo frame ID; giữ index người gốc và khớp không finite bằng null; ghi frame lỗi/missing/extra vào `mapping_audit.json`. Dùng tracker dispatcher/config giống loader và ngưỡng confidence chung.
- Ghép selected OpenPose person với ROMP bằng 12 cặp khớp tương ứng trong image XY, median distance / bbox diagonal, ngưỡng 0,18 và ambiguity margin 0,025. Đây là heuristic association, chưa phải nhãn child identity đã xác minh. Nếu thiếu hoặc mơ hồ, không tự tô người khác làm TARGET.
- Tại frame 25 trong ảnh: selected OpenPose index 3 được ghép với ROMP index 4, score khoảng 0,0857. Viewer cũ đã tô ROMP index 3. Không thể dùng con số này làm xác suất đúng.
- Namespace review cũ được giữ. Bản nháp không có schema mới được đánh dấu cần xem lại; review mới lưu schema version. Dữ liệu gốc và loader/model huấn luyện không bị thay đổi.

Mã mới: [đọc và ghép dữ liệu ROMP](../../BiLSTM/scripts/romp_review_data.py), [renderer cơ thể 3D](../../BiLSTM/scripts/tracking_review_3d_renderer.js), [kiểm thử mapping](../../BiLSTM/tests/test_romp_review_mapping.py).

Validation: 8 kiểm thử mapping/parser/tracking đạt; kiểm tra cú pháp JavaScript và `make check` đạt. Chrome headless đã chạy chính diện/bên/trên, ba chế độ hiển thị, toàn bộ 100 frame với tất cả người của clip trong ảnh; không có lỗi JavaScript. Kiểm tra projection xác nhận đầu trên chân, lưới dưới tâm, mesh hữu hạn và frame chưa match không tự chọn người. Đây là kiểm tra renderer/association, chưa xác minh chất lượng pose ROMP hoặc độ chính xác chọn trẻ.

Sau khi xuất lại 120 clip / 27.374 frame: 24.306 frame có match đạt ngưỡng heuristic, 3.006 chưa match và 62 ambiguous; không còn selected ROMP index vượt phạm vi. Kết quả này đo trạng thái association, không phải tỷ lệ chọn đúng trẻ. Chi tiết và source-code hashes nằm trong `BiLSTM/artifacts/tracking_review_3d/mapping_audit.json`.

## 8. Sửa mất hình và nâng chất lượng bề mặt (viewer v3)

Người dùng tiếp tục báo mất hình và chất lượng mô phỏng chưa đẹp. Kiểm tra toàn bộ payload cho thấy cả 27.374 frame đều có ROMP candidates; không có frame rỗng nguồn hoặc pose thiếu core trong snapshot này. Có 3.068 frame không ghép chắc được người. Chế độ mặc định chỉ xem TARGET của v2 đã làm các frame này trống dù nguồn vẫn có hình.

V3 mặc định hiển thị tất cả ứng viên. Khi lọc TARGET hoặc raw person index mà người đó không có mặt, viewer tự mở toàn cảnh và ghi rõ chưa ghép/vắng người; không gán ứng viên khác thành trẻ. Camera tự căn bounds theo góc nhìn, có khoảng đệm cho bề mặt, và canvas thích ứng kích thước/DPI. Pose thiếu một phần vẫn dựng được các bộ phận có căn cứ; nếu chưa dựng được surface thì hiện skeleton nguồn.

Bề mặt chuyển sang WebGL offline với depth buffer, anti-aliasing và normals nội suy, theo các nguyên tắc trong [WebGL lighting](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Lighting_in_WebGL) và [depth testing](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/depthFunc). Thân dùng loft mượt qua cột sống; đầu, tay, bàn tay/chân và khớp được bo tròn, phối màu áo/quần, bổ sung vùng chậu và nối cổ để giảm khe hở. Ánh sáng key/fill và highlight thay cho tô phẳng từng tam giác. Không sửa tọa độ nguồn hoặc nội suy chuyển động qua các raw person index. Context mất/hồi phục có đường render tương thích.

Đã kiểm tra logic visibility của đúng 120 trang / 27.374 frame: default không bỏ ứng viên nguồn, TARGET chưa match và person index vắng đều có fallback toàn cảnh; 3.068 frame từng bị ẩn được hiển thị lại. Chrome kiểm tra WebGL thực, các góc trước/bên/trên, ba chế độ hiển thị, perspective và các frame chưa match. Clip `mfs_40683_D8_000_i` frame 0 từng trống đã được kiểm tra trực quan. Kiểm thử mới `test_3d_renderer_visibility.py` kiểm tra thêm partial pose, orbit clipping và không dựng cơ thể từ pose suy biến; nhóm kiểm thử renderer + mapping đạt 7/7.
