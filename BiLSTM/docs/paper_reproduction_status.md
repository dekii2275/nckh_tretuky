# Trạng thái tái lập bài báo

Đối chiếu ngày 12/09/2026. Nguồn chính là PDF người dùng cung cấp,
DOI [10.1109/ICCA66035.2025.11431081](https://doi.org/10.1109/ICCA66035.2025.11431081),
mục III.B–D, Bảng II và mục IV. SHA-256 của PDF được ghi trong
`artifacts/paper_repro/protocol_audit.json`.

## Kết luận hiện tại

`configs/paper_repro.yaml` khớp các thông số kiến trúc và huấn luyện được
công bố. Chưa xác nhận được tái lập chính xác thí nghiệm tác giả vì thiếu
quy trình tạo mẫu, danh sách dữ liệu và split gốc. Không có cơ sở để cam kết
đạt 94,84% từ PDF hiện có.

Tìm kiếm theo tên bài, DOI, tên tác giả và GitHub chưa tìm được mã nguồn có
thể xác nhận là của bài này. Trang IEEE không truy cập được qua công cụ duyệt
web trong lần kiểm tra; vì vậy không kết luận rằng tác giả chưa từng công bố
code. [Trang tác giả tại UTT](https://recherche.utt.fr/research-directory/mohammad-kassir)
cung cấp địa chỉ liên hệ. Người dùng cũng xác nhận chưa có code/notebook.

[Repository MMASD chính thức](https://github.com/Li-Jicheng/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis)
cung cấp dataset và mô tả OpenPose 25 joints; đây không phải implementation
của bài BiLSTM. Tài liệu của nghiên cứu khác không được dùng để lấp chỗ trống
trong thiết lập của tác giả.

## Những phần đã xác nhận

Input 60 × 50; 25 joints với x/y; chuẩn hóa theo MidHip và khoảng cách hai
vai; BiLSTM 2 lớp, hidden 128 mỗi chiều; attention 4 heads; residual và
LayerNorm; temporal mean pooling; Linear 256→64, ReLU, dropout 0,3 và 10
outputs. Adam, learning rate 0,001, batch 32, cross-entropy, tối đa 100 epoch,
early stopping patience 10, train/validation mục tiêu 80/20 theo session.

Các kiểm tra YAML không tự chứng minh hành vi runtime. Mô hình và đường đi
dataset được đối chiếu với code và kiểm thử riêng. Code trả về logits cho
CrossEntropyLoss; việc không thêm softmax trước loss không phải sai kiến trúc
phân loại khi lấy argmax.

## Những phần chưa được công bố đủ rõ

| Điểm cần xác minh | Giả định hiện tại | Bằng chứng cần từ tác giả |
|---|---|---|
| Tạo mẫu | Một clip → lấy mẫu đều/nội suy thành 60 frame | Hàm tạo sequence, stride, FPS, xử lý clip ngắn/đuôi |
| Danh sách clip | 1.374 clip OpenPose có JSON, bỏ Frog Pose | Inventory cuối cùng, tiêu chí loại clip |
| Split | `subject_id::session_id`, seed 42, chọn nhóm có phủ lớp | Định nghĩa session và danh sách train/validation |
| Chọn người | Temporal tracker, ưu tiên skeleton nhỏ | Quy tắc chọn/tracking trẻ hoặc dùng nhiều người |
| Keypoint thiếu | Threshold 0,1, nội suy, fallback vai <5 px | Quy tắc preprocessing gốc |
| Huấn luyện phụ | Monitor accuracy, internal dropout 0, weight decay 0 | Monitor, dropout, optimizer defaults và phiên bản thư viện |
| Đánh giá | Một dự đoán cho mỗi clip | Đơn vị đánh giá và confusion matrix dạng số đếm |

Không giảm số clip xuống 1.315 hoặc 1.202 chỉ để khớp một con số: bài mô tả
1.315 clip của dataset gốc, không cung cấp inventory cuối cùng của thí nghiệm
10 lớp. 1.202 không phải số mẫu thực nghiệm được xác nhận trong bài.

## Kiểm tra dữ liệu hiện có

Train có 1.091 clip / 201.384 frame / 55 nhóm subject-session; validation
có 283 clip / 53.290 frame / 17 nhóm. Tỷ lệ thực tế là 79,40%/20,60%, do
không tách nhóm. Không có clip, path hoặc nhóm subject-session trùng giữa
hai tập. Split không phải cross-subject.

Chỉ để mô tả độ nhạy của định nghĩa mẫu, audit tính số cửa sổ 60 frame liên
tiếp có thể tạo, bỏ đoạn ngắn và phần đuôi:

| Cách tạo mẫu giả định | Train | Validation |
|---|---:|---:|
| Mỗi clip một mẫu — đang dùng | 1.091 | 283 |
| Window 60, stride 60 | 2.970 | 779 |
| Window 60, stride 30 | 5.395 | 1.412 |
| Window 60, stride 1 | 137.148 | 36.665 |

Đây là số đếm từ manifest, không phải thiết lập tác giả, không tạo dữ liệu
window và không dùng accuracy để chọn stride. Nếu sau này xác nhận dùng
window, phải chia clip/session trước rồi mới tạo window riêng trong mỗi tập;
không ghép qua ranh giới clip hoặc người.

Hình 5 của bài có tỷ lệ nhỏ như 0,01%. Nếu là số đếm dự đoán chuẩn hóa theo
hàng, một tập validation chỉ vài trăm clip không tạo được tỷ lệ này. Đây là
lý do hỏi tác giả về số lượng mẫu/đơn vị đánh giá; cũng có thể có cách tổng
hợp khác hoặc vấn đề trình bày. Chưa đủ bằng chứng để kết luận sliding window
hay leakage.

## Những cải thiện cho lần chạy tiếp theo

- Checkpoint mới lưu cấu hình đã resolve, SHA-256 các manifest/split và code
  Python trong `src`. Hash CSV không phải hash nội dung skeleton gốc.
- Evaluation từ chối khi cấu hình đầu vào/mô hình hoặc CSV thay đổi so với
  checkpoint có hash. Khác code được báo riêng. Checkpoint cũ không có hash
  được ghi rõ là chưa xác minh split lịch sử, không được gán xác nhận hồi tố.
- Mỗi lần đánh giá xuất dự đoán từng clip, confusion matrix dạng số đếm và
  phần trăm, giúp xác định chính xác mẫu số của metric.
- `best_val_macro_f1` ở history mới lưu F1 tại checkpoint được chọn; trước
  đây tên trường này bị dùng để lưu accuracy khi monitor là accuracy.
- Pipeline có `--reuse-splits` để dùng nguyên CSV đã chốt. Không dùng
  `prepare_manifest.py` lên split tác giả cung cấp nếu muốn giữ split đó.

## Chạy lại kiểm chứng

Từ thư mục gốc repository, trong môi trường `mmasd-bilstm`:

```bash
python BiLSTM/scripts/audit_paper_protocol.py --config BiLSTM/configs/paper_repro.yaml
python -m pytest -q BiLSTM/tests
python -m BiLSTM.src.evaluate \
  --config BiLSTM/configs/paper_repro.yaml \
  --checkpoint BiLSTM/checkpoints/paper_repro/best.pt \
  --split val \
  --output-dir BiLSTM/artifacts/paper_repro/verified_evaluation
```

Đánh giá lại checkpoint epoch 32 trên toàn bộ 283 clip cho accuracy 67,8445%
và macro F1 65,7509%, bằng các metric phân loại đã lưu trước đó. Chưa tái lập
được 94,84%. Kết quả mới nằm riêng trong `verified_evaluation/`.

Để huấn luyện sau khi đã có thiết lập cần kiểm tra, dùng một YAML kế thừa
`paper_repro.yaml`, giữ split đã chốt và đặt `checkpoints_dir`/`runs_dir` sang
thư mục mới để giữ checkpoint cũ. Pipeline hỗ trợ:

```bash
python BiLSTM/scripts/run_full_pipeline.py --config PATH_TO_NEW_CONFIG --reuse-splits
```

Bước cần thông tin bên ngoài là xác nhận các chi tiết chưa công bố.
[Bản email soạn sẵn](author_reproduction_request.md) hỏi trực tiếp những điểm
này. Bản này chỉ lưu cục bộ, chưa gửi.
