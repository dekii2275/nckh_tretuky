# Kế hoạch 7 ngày: pipeline đánh giá, phát hiện và hiệu chỉnh skeleton

**Ngày lập:** 14/10/2026. D1 tính từ ngày bắt đầu thực tế.
**Trạng thái:** kế hoạch triển khai theo mục tiêu nghiên cứu người dùng đã chốt; chưa triển khai pipeline mới.
**Dữ liệu đã xác nhận:** chưa có video trẻ để nghiên cứu; tuần đầu dùng skeleton MMASD có sẵn.
**Cơ sở:** [kế hoạch nghiên cứu](KE_HOACH_NGHIEN_CUU_SKELETON_RELIABILITY.md), [kế hoạch demo trước](KE_HOACH_7_NGAY_MMASD_RELIABILITY_DEMO.md) và [rà soát viewer 3D](../reports/RA_SOAT_MO_PHONG_3D_VA_CHUAN_HOA_DU_LIEU.md).

## 1. Mục tiêu nghiên cứu và sản phẩm cuối tuần

Mục tiêu đã chốt: **Nghiên cứu và xây dựng một pipeline tự động đánh giá, phát hiện và hiệu chỉnh các lỗi trong dữ liệu khung xương của trẻ em từ video, với trọng tâm kiểm chứng trên dữ liệu trẻ ASD, nhằm nâng cao độ tin cậy của dữ liệu chuyển động phục vụ phân tích hành vi.**

Bốn mục tiêu: (1) khảo sát/chuẩn hóa skeleton; (2) đánh giá chất lượng và phát hiện lỗi; (3) đề xuất hiệu chỉnh bằng interpolation, filtering, outlier correction và hình học khi đủ căn cứ; (4) đánh giá định lượng/trực quan trước–sau theo activity/joint/điều kiện có dữ liệu hỗ trợ. ASD là miền kiểm chứng chính. Schema và viewer phục vụ phương pháp; kết quả nhận dạng là kiểm chứng bổ sung.

Kết quả tuần đầu gồm pipeline nghiên cứu và web kiểm chứng chạy cục bộ: chọn clip MMASD → xem dữ liệu gốc → xem cảnh báo theo frame/khớp → so sánh bản xử lý → lưu review → xem thống kê → xuất dữ liệu và hồ sơ xử lý. Từ D1 phải có luồng xem dữ liệu thật trong trình duyệt. Bước video → pose extraction chưa được kiểm chứng trong đợt này.

Đóng góp cần bàn giao gồm:

1. Đặc tả dữ liệu v0.1 và data dictionary: nguồn, người, thời gian, khớp, hệ tọa độ, chất lượng và lịch sử biến đổi.
2. Pipeline chuyển dữ liệu MMASD vào đặc tả đó, đánh giá/phát hiện và hiệu chỉnh có điều kiện; có baseline nội suy, một filter, variant outlier correction và cấu hình/nguồn để chạy lại.
3. Web demo 2D/3D, so sánh raw/processed, timeline, dashboard và review theo đoạn frame/khớp.
4. Bản dữ liệu xử lý có phiên bản, manifest, mask và transform; có thể đọc lại ngoài ứng dụng web.
5. Báo cáo thử nghiệm trên mẫu đã khóa, gồm detection có nhãn phù hợp, sai số khôi phục, bảo toàn chuyển động, ablation và ảnh hưởng tới nhận dạng nếu checkpoint có đủ provenance.

Khả năng áp dụng cho trẻ em nói chung là hướng mở rộng cần dữ liệu kiểm chứng. Tuần đầu chưa đủ để công bố một chuẩn phổ quát hoặc bộ skeleton ground truth.

## 2. Phạm vi và những điều chỉnh so với nội dung tham khảo

| Nội dung | Quyết định cho tuần đầu |
|---|---|
| Dataset | MMASD là nguồn triển khai. Demo 3–5 clip; pilot 20–50 clip được chọn theo activity, subject/session, độ dài và mức missing. Khóa manifest và tách tuning/evaluation theo nhóm trước khi chọn threshold/filter. |
| 2D/3D | BODY_25 2D là nhánh đo và xử lý chính. ROMP 3D phục vụ xem/review; giữ hệ nguồn và ghi trạng thái đơn vị chưa xác minh. Không tạo Z từ dữ liệu 2D. |
| Viewer hiện có | Tái sử dụng parser, tracker dispatcher, join frame ID, association và renderer đã có; bọc renderer trong component trước khi cân nhắc chuyển sang React Three Fiber. |
| Quality score | Hiển thị từng metric và lý do cảnh báo; chưa có total score, good/bad %, hoặc xác suất pose đúng. |
| Correction | Bản xử lý v0.1 nội suy tuyến tính gap ngắn có quan sát hợp lệ ở hai đầu, cùng track. Khoảng dài, biên chuỗi và danh tính mơ hồ để thiếu. |
| Filtering/outlier | Một filter là baseline cần có sau nội suy. Thử correction cho temporal outlier khi có tín hiệu phối hợp; không loại chuyển động chỉ vì nhanh. Đo mất biên độ/độ trễ; không mặc định ép bone constraint trong 2D. |
| Video/reference | Bản cục bộ thiếu RGB gốc. Panel tham chiếu dùng skeleton/metadata có sẵn; không dựng video thay thế. |
| Annotation | Lưu nhận xét/cảnh báo và trạng thái chưa xác định. Không gán occlusion, sai danh tính hay sai pose đã xác minh từ skeleton đơn thuần. |
| DREAM/DRYAD | Schema có chỗ cho topology/đơn vị khác; chưa triển khai thêm loader/UI trong tuần đầu, không gộp label space. |
| Video → pose extraction | Thuộc phạm vi đề tài đầy đủ, chờ video trẻ phù hợp. Tuần đầu dùng output pose estimator có sẵn, không công bố đã kiểm chứng toàn bộ luồng từ video. Upload UI là feature giai đoạn sau. |

Viewer đã có surface minh họa từ khớp, không phải mesh SMPL được phục hồi. Association OpenPose–ROMP là heuristic; lưu matched/unmatched/ambiguous. Dữ liệu ROMP giữ image XY và Z chưa xác minh nên chưa dùng để tính khoảng cách, góc hoặc độ dài xương 3D mang đơn vị vật lý.

## 3. Kiến trúc triển khai

Stack lựa chọn: Next.js + TypeScript cho giao diện, FastAPI cho đọc artifact và lưu review, Python/NumPy cho xử lý offline; JSON/NPZ cho dữ liệu, SQLite cho review. Chỉ thêm SciPy nếu có bộ lọc cần dùng. Chọn một thư viện biểu đồ tại D1. Khóa dependencies trong lockfile và ghi môi trường thực tế.

React Three Fiber là lựa chọn cho viewer React khi cần phát triển tiếp; trong tuần này ưu tiên tích hợp renderer WebGL sẵn có để giảm công viết lại. Frontend không đọc format gốc dataset, không tự chạy một pipeline chuẩn hóa khác.

```text
MMASD gốc
  → parser/tracking hiện có
  → adapter đặc tả v0.1
  → quality trên selected raw
  → xử lý thiếu / normalization có truy vết
  → artifact có phiên bản
  → FastAPI → Next.js: browser / viewer / timeline / dashboard
                    ↘ SQLite: review theo clip/frame/joint
```

Vị trí dự kiến: `apps/skeleton_workbench/frontend/`, `apps/skeleton_workbench/backend/`, `tools/skeleton_workbench/` và `BiLSTM/artifacts/standardization_v0_1/`. Python mới gọi module dữ liệu hiện có khi phù hợp. Đây là đường dẫn dự kiến, chưa có ứng dụng tại các thư mục đó.

API tối thiểu: danh sách clip và bộ lọc; đọc clip theo processing version; đọc quality/events; đọc/ghi review; xuất manifest và dữ liệu xử lý. Clip dài được đọc theo đoạn frame, không tải cả dataset trong một request. Threshold chỉ đổi cờ cảnh báo trên metric đã tính; thay policy xử lý phải tạo variant mới, không sửa artifact đang xem.

Tài liệu triển khai chính thức: [Next.js](https://nextjs.org/docs/app/getting-started/installation), [FastAPI](https://fastapi.tiangolo.com/tutorial/), [React Three Fiber](https://r3f.docs.pmnd.rs/getting-started/introduction). Kiểm tra tương thích môi trường và thư viện ở D1; chưa chốt số phiên bản bằng tài liệu này.

## 4. Đặc tả dữ liệu v0.1

Một cấu trúc metadata chung nhưng tọa độ/topology/nhãn được giữ theo modality và nguồn. Không ép BODY_25 và SMPL24 thành cùng graph.

| Nhóm | Trường tối thiểu |
|---|---|
| Định danh | `schema_version`, `dataset_id`, `clip_id`, `subject_id`, `session_id`, `activity`, source path/hash, config/code hash |
| Thời gian | source frame ID, timestamp khi có, `fps`, `fps_status`; FPS chưa biết để null, không mặc định 30 |
| Hình học | modality, topology ID, joint names/edges, số chiều, coordinate space, unit theo trục, trạng thái xác minh |
| Người | source candidate index theo modality, selected index, track ID khi có, match status/score; array index không phải danh tính |
| Khớp | tọa độ, confidence nguồn hoặc null, observed/valid/imputed mask, missing reason, repair method |
| Biến đổi | root, scale, rotation nếu dùng, fallback, processing version; liên kết model frame về source frame |
| Review | clip/frame interval/joint/modality, label, uncertainty, note, artifact version/hash, thời điểm và reviewer khi có |

Giữ các lớp `raw → selected → processed → model_input`. Bản demo chỉ xuất `model_input` nếu đã khóa pipeline/checkpoint; không giả định mọi experiment nhận cùng representation. Confidence không có để null; confidence của khớp nội suy không được tự nâng. Review là artifact riêng, không tự sửa tọa độ hoặc override track.

Phân biệt hai việc: **chuẩn hóa cấu trúc dữ liệu** để đọc/truy vết thống nhất, và **normalization tọa độ** để tạo đầu vào cho tác vụ. Normalization 2D thử theo root/scale hợp lệ, ghi fallback và giữ transform nghịch. Viewer so raw/processed ở cùng hệ bằng inverse transform hoặc hiển thị rõ từng hệ. Không chuẩn hóa hướng mặc định làm mất tín hiệu xoay/di chuyển.

## 5. Quality và xử lý phiên bản đầu

Metric trên selected raw, trước điền thiếu và resampling:

- Missing-joint ratio và độ dài missing run; khai báo mẫu số và validity policy.
- Confidence trung bình/median và low-confidence ratio; trường không có confidence để NA.
- Displacement/second difference như chỉ báo chuyển động đột ngột. Nếu không có FPS đã xác minh, dùng đơn vị mỗi frame; chuyển động thật cũng có thể tạo spike.
- Biến thiên độ dài đoạn xương chiếu 2D, dùng cặp khớp hợp lệ; không diễn giải thành độ dài xương vật lý vì thay đổi góc nhìn cũng làm biến thiên.
- Số người, target missing, thay đổi candidate index và center jump; gọi là tracking indicators, không gọi verified ID switch.
- Trạng thái association 2D/3D và fallback của preprocessing.

Lưu metric liên tục và events gồm frame/joint, giá trị, threshold/config và reason. Ngưỡng cảnh báo là heuristic, có sensitivity check; chỉnh trên UI không biến cảnh báo thành ground truth.

Xử lý v0.1: tạo validity mask → phát hiện và đánh dấu outlier candidates → nội suy gap ngắn được kẹp hai đầu trên cùng track → một filter trên đoạn hợp lệ liên tục → normalization tùy variant → xuất transform/provenance. Variant outlier correction chỉ chuyển candidate thành missing để tái dựng khi có bằng chứng phối hợp và giới hạn rõ; không sửa riêng từ displacement lớn. Giữ mask phân biệt observed, removed, imputed và filtered. Không nối qua track boundary, gap dài hay biên chuỗi. `processed` là tên mặc định trên giao diện; nếu dùng `Corrected`, phải ghi rõ là bản ước lượng.

So sánh B0 raw/no repair, B1 interpolation, B2 interpolation + filter, B3 conditional outlier correction + B2 trên cùng dữ liệu/seed; không chọn pipeline cuối chỉ từ hình đẹp nhất. Baseline filter có thể đối chiếu [Savitzky–Golay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) hoặc [1€ Filter](https://gery.casiez.net/1euro/); chốt một phương pháp theo sampling đã kiểm tra. Nếu FPS chưa xác minh thì tham số/metric dùng frame, không giả làm Hz hay vận tốc theo giây.

## 6. Lịch thực hiện và điều kiện hoàn tất mỗi ngày

| Ngày | Việc chính | Đầu ra nghiệm thu |
|---|---|---|
| **D1 — Nguồn và protocol** | Audit dữ liệu, khảo sát phương pháp liên quan; khóa pilot/tuning/evaluation, schema và config; scaffold web/API, tích hợp viewer sẵn có. | Browser phát clip thật; có `protocol.md`, manifest theo nhóm, bảng phương pháp liên quan và schema draft. Không chờ checkpoint để bắt đầu. |
| **D2 — Quality và phát hiện** | Tính missing/confidence/gap, temporal/geometric/tracking indicators; xuất events và nối timeline. Chọn tiêu chí outlier candidates trên tuning. | Click event tới đúng frame/joint; lưu giá trị/reason/config; có NA/mẫu số. Geometry 2D chỉ là proxy, tracking ambiguity không bị tự sửa. |
| **D3 — Baseline hiệu chỉnh** | Nội suy ngắn và một filter; thử conditional outlier correction; masks và normalization variant; nối raw/processed/compare. | B0/B1/B2 chạy được, B3 có rule/config hoặc ghi rõ chưa xong; compare cùng người/frame/hệ; gap dài còn thiếu, export đọc lại được. |
| **D4 — Viewer và review** | Hoàn thiện camera, chọn joint, tên khớp, legend, loading/empty/error; lưu review cấp đoạn frame/khớp qua API/SQLite. | Orbit/zoom/reset; frame ambiguous vẫn xem được nguồn; review tồn tại sau reload và export/import giữ version. Surface không che cảnh báo khớp. |
| **D5 — Benchmark và phân bố lỗi** | Chạy pilot; tạo missing runs/spikes có nhãn vị trí, mức độ/seed; dashboard theo activity/joint/session/số người; khóa data dictionary và configs. | Manifest corruption/reference, quality tables và artifact v0.1; đủ dữ liệu chạy so B0–B3, không tuning trên nhóm evaluation. |
| **D6 — Đánh giá và ablation** | So detector/correction theo cùng mẫu/seed; recovery, coverage và motion preservation; paired differences theo clip/nhóm; inference là bổ sung. | Bảng trước–sau/ablation, uncertainty và failure cases; detection metrics chỉ khi có nhãn phù hợp. Không kết luận significancy nếu pilot chưa đủ bằng chứng. |
| **D7 — Bàn giao** | Không thêm feature lớn; sửa lỗi, smoke test browser/API/export, viết README và báo cáo, chuẩn bị 3–5 clip trình bày. | Chạy trọn flow 3–5 phút từ lệnh đã ghi; bàn giao schema, artifact, review, report và giới hạn; các mục chưa làm được ghi rõ. |

## 7. Đánh giá để demo có giá trị nghiên cứu

Không dùng missing giảm sau nội suy hay jitter giảm sau smoothing làm bằng chứng sửa đúng pose. Báo cáo riêng **thiếu ở nguồn**, **đã điền** và **còn thiếu**; không đổi mẫu số để tăng tỷ lệ hợp lệ.

Đánh giá khôi phục: chọn điểm vốn hợp lệ, che theo đoạn có seed, rồi so vị trí khôi phục với quan sát được giữ lại. Nếu normalization có dùng tọa độ, fit lại transform từ phần còn quan sát sau khi che để tránh rò rỉ; lỗi tính trong hệ thống nhất, với scale tham chiếu được khai báo. Thử gap ngắn/dài và baseline giữ giá trị gần nhất khi có căn cứ. Quan sát giữ lại vẫn là output pose estimator, không phải pose ground truth. Báo cáo theo joint/activity/gap length và coverage.

Đánh giá bảo toàn chuyển động: kiểm tra biên độ, quỹ đạo, thời điểm cực trị và displacement trên đoạn vốn quan sát được; xem các clip chuyển động nhanh để tránh làm mất tín hiệu hành vi. So normalization và interpolation bằng các variant tách biệt khi cần xác định tác động.

Đánh giá phát hiện: trên corruption có nhãn vị trí, đo precision/recall/F1 theo loại/severity và false alarms trên các vị trí tham chiếu đã khai báo. Dữ liệu nền có thể đã chứa lỗi thật nên precision này chưa đo độ chính xác phát hiện lỗi tự nhiên. Review skeleton không thay thế annotation RGB về sai pose/tracking/occlusion.

So B0–B3 theo paired differences, effect size và khoảng tin cậy theo subject/session khi số nhóm cho phép. Tuning/evaluation tách theo nhóm; frame liên tiếp không được xem là mẫu độc lập. Pilot nhỏ hoặc ít subject phải ghi kết quả thăm dò. Không mặc định pipeline phải cải thiện mọi chỉ số hay mọi hoạt động.

Đánh giá nhận dạng là bổ sung: khóa checkpoint/config/split/prediction trước khi chạy, không chặn quality/correction nếu provenance chưa rõ. Đối chiếu bất nhất `improved_v1`/Tracker V2/`romp_guided` trước khi chọn. Giữ tracking, windowing, aggregation và clip set đồng nhất; nhánh xử lý mới có config riêng và không được vô tình chạy thêm preprocessing cũ lần nữa. Đây là thử độ tương thích với checkpoint cố định; không đồng nghĩa mô hình được huấn luyện trên đầu vào chuẩn hóa mới.

20–50 clip dùng để pilot xử lý; không suy rộng thống kê cho cả dataset. Nếu đánh giá nhận dạng như kết quả chính, dùng toàn bộ validation hợp lệ của checkpoint đã khóa và công bố clip thiếu/bị loại. Nếu chỉ chạy inference trên pilot, ghi rõ kết quả thăm dò. Không gọi validation là test độc lập.

## 8. Mức ưu tiên và điều kiện thu hẹp

**P0 phải có:** protocol/literature comparison, schema/data dictionary, adapter MMASD, quality/events, nội suy ngắn và một filter với masks, benchmark missing/spikes, đánh giá detection/recovery/motion preservation và so baseline. Viewer tận dụng 2D/3D hiện có, play/seek/compare, timeline, dashboard đơn giản và export là công cụ kiểm chứng.

**P1 cố hoàn thành:** variant conditional outlier correction đầy đủ, inspector tương tác sâu hơn, review qua SQLite, heatmap theo joint/activity và inference đối chứng khi checkpoint sẵn sàng. Review chưa kịp vào SQLite thì xuất/import JSON có version là phương án bàn giao tối thiểu. B3 chưa xong phải ghi rõ, không mô tả đã kiểm chứng đủ mọi loại correction.

**P2 giai đoạn sau:** video → pose extraction khi có nguồn trẻ phù hợp; RGB annotation để xác minh pose/tracking/occlusion; loader DREAM/DRYAD; hình học/cơ thể với 3D/units đã xác minh; tổng điểm chất lượng đã hiệu chỉnh; mesh SMPL chuẩn, model reliability-aware và deployment công khai.

Cuối D1 nếu tích hợp renderer vào React chưa chạy, nhúng viewer hiện có bằng adapter/postMessage cho clip/frame/variant và tiếp tục hoàn thành pipeline; chưa viết lại renderer. Cuối D3 nếu compare chưa xong, giảm chart/inspector/smoothing. Cuối D5 nếu số liệu chưa chạy toàn bộ pilot, báo coverage thực tế; ưu tiên export và đánh giá trên dữ liệu thật hơn mở rộng feature.

Kế hoạch giả định làm việc tập trung và dữ liệu/môi trường cục bộ dùng được. Đây là mục tiêu có điều kiện nghiệm thu, không cam kết rằng toàn bộ nội dung tham khảo chắc chắn hoàn thành trong bảy ngày.

## 9. Checklist bàn giao

- [ ] Schema v0.1 và data dictionary, có quy tắc unknown/NA/topology/units.
- [ ] Protocol, literature comparison, manifest tuning/evaluation, dependency lock và config/source/code hashes.
- [ ] UI xem clip thật, timeline và raw/processed compare đồng bộ.
- [ ] Quality tables và processed artifact có mask/transform, đọc lại được.
- [ ] Review export/import có phiên bản và uncertainty.
- [ ] Báo cáo pilot, detection/recovery evaluation, motion preservation, ablation và recognition status.
- [ ] Kiểm tra lỗi frame thiếu, nhiều người/ambiguous, khớp NaN, gap dài/biên/switch, export round-trip và review persistence.
- [ ] README chạy lại, demo 3–5 phút, hạn chế và công việc tiếp theo.
