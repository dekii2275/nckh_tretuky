# Phát triển hướng nghiên cứu chất lượng skeleton

Nhánh làm việc: **`research/skeleton-reliability`**. Mục tiêu và tiến độ nằm trong [trang tiến độ](reports/TIEN_DO_NGHIEN_CUU.md); [báo cáo pipeline](reports/RA_SOAT_VA_KHOA_PIPELINE_MMASD.md) giải thích điểm đo raw và các bất nhất lịch sử.

## Chuẩn bị và kiểm tra không cần dữ liệu riêng

```bash
git clone --branch research/skeleton-reliability https://github.com/dekii2275/nckh_tretuky.git
cd nckh_tretuky
make check
python3 -m pip install numpy PyYAML pytest torch ruff
make test-audit
ruff check BiLSTM/scripts/audit_pipeline_lock.py BiLSTM/tests/test_pipeline_lock.py
ruff format --check BiLSTM/scripts/audit_pipeline_lock.py BiLSTM/tests/test_pipeline_lock.py
```

`make check` dùng standard library. `make test-audit` dùng dữ liệu tổng hợp trong thư mục tạm: kiểm tra giữ X/Y/confidence, missing target, cấu hình, hash, inventory frame, artifact và chống ghi đè snapshot. Các kiểm thử này không cần MMASD, checkpoint, GPU hoặc mạng sau khi đã cài dependencies.

[GitHub Actions](../.github/workflows/skeleton-reliability.yml) tự chạy kiểm tra cú pháp/liên kết, lint/format và nhóm kiểm thử audit khi push lên nhánh nghiên cứu hoặc tạo pull request. Bộ kiểm thử đầy đủ chạy bằng `make test` khi đã chuẩn bị dependencies và dữ liệu cục bộ của hai pipeline.

## Chạy audit với dữ liệu cục bộ

Cần dữ liệu MMASD, các checkpoint và history tương ứng trên máy. Chúng không đi kèm Git. Chuẩn bị chúng theo [README dự án](../README.md) và [README MMASD](../BiLSTM/README.md).

```bash
# Khi chưa có thư mục output v0.2:
make audit-pipeline

# Kiểm tra snapshot v0.2 hiện có, không ghi lại file:
make verify-pipeline

# Tạo một mốc mới, không ghi đè snapshot cũ:
python3 BiLSTM/scripts/audit_pipeline_lock.py \
  --output-dir BiLSTM/artifacts/pipeline_audit_next
```

Cấu hình mặc định là [skeleton_audit_v0_2.yaml](../BiLSTM/configs/skeleton_audit_v0_2.yaml). Script tách các nhiệm vụ thành `observe_raw`, `reconcile_experiments`, `reconcile_stacking`, `smoke_pipeline`, `verify_snapshot` và `write_snapshot`. Đường dẫn clip được dựng từ data root trong config, thay vì phụ thuộc đường dẫn tuyệt đối trong manifest cũ. Parser/tracker/preprocessing huấn luyện hiện có được gọi lại, không sao chép thuật toán sang một pipeline khác.

Snapshot v0.2 lưu hash code/input, các YAML kế thừa, raw JSON của ba clip smoke và hash artifact đầu ra; đồng thời ghi giờ tạo theo múi giờ Việt Nam và phiên bản thư viện. Việc xác minh snapshot cục bộ cần đúng input và phiên bản môi trường đã ghi. Trên bản clone chưa có dữ liệu/checkpoint/history, chạy `make check` và `make test-audit` trước; chưa thể verify toàn snapshot.

## Quản lý phiên bản và cập nhật tiến độ

Công cụ pilot mới: xem [hướng dẫn chọn clip](reports/CONG_CU_CHON_PILOT_MMASD.md). Dùng `make pilot-catalog` để đo dữ liệu, `make pilot-selector` để mở giao diện và `make test-pilot` để kiểm tra bằng dữ liệu tổng hợp. Không sửa snapshot audit v0.2 khi chỉ bổ sung giao diện/logic chọn mẫu; catalogue ghi riêng hash của code đo và config, hồ sơ xuất ghi thêm hash công cụ chọn.

- Giữ [snapshot v0.1](../BiLSTM/artifacts/pipeline_audit_v0_1/pipeline_lock.json) như bằng chứng lịch sử. Bản script ban đầu được lưu tại [audit_pipeline_lock.py.txt](../BiLSTM/artifacts/pipeline_audit_v0_1/audit_pipeline_lock.py.txt). Code hiện tại đã thay đổi nên hash script của v0.1 sẽ báo khác; không sửa hash cũ để làm verify đạt.
- Bản chuẩn hóa dùng [snapshot v0.2](../BiLSTM/artifacts/pipeline_audit_v0_2/pipeline_lock.json). Khi đổi code/config/input ảnh hưởng pipeline, tạo snapshot phiên bản mới và ghi lý do; hash chỉ phủ các input được liệt kê, chưa phủ toàn MMASD.
- Khi sửa code audit, chạy lint/format và `make test-audit`; khi sửa parser/tracker/preprocessing, chạy thêm các kiểm thử liên quan và kiểm tra dữ liệu thật trước khi lập snapshot mới.
- Sau mỗi mốc, cập nhật [tiến độ và nhật ký](reports/TIEN_DO_NGHIEN_CUU.md), README và báo cáo có liên quan. Phân biệt kết quả đã đo, kiểm thử kỹ thuật, cảnh báo heuristic và nhãn đã được người xác minh.
- Commit code/config/tài liệu cùng artifact tổng hợp cần thiết. Dữ liệu gốc, checkpoint, môi trường và HTML viewer lớn tiếp tục được quản lý cục bộ theo `.gitignore`.
