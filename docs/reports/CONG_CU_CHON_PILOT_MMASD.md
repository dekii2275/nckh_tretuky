# Công cụ chọn pilot MMASD trực quan

**Ngày cập nhật:** 18/10/2026.

Đã có ứng dụng cục bộ để đề xuất, xem, điều chỉnh và chốt bộ clip nghiên cứu. Công cụ dùng parser và Tracklet V2 đã khóa, quan sát trước nội suy, normalization, smoothing và sampling. Chỉ số chất lượng hỗ trợ chọn mẫu; chưa có bộ nhãn xác minh pose/danh tính.

## Mở công cụ

Chạy từ thư mục gốc dự án:

```bash
# Chỉ cần tạo lại khi chưa có catalogue hoặc dữ liệu/code đo đã đổi:
make pilot-catalog

# Catalogue trên máy làm việc hiện tại đã có:
make pilot-selector
```

Mở `http://127.0.0.1:8787`. Dừng máy chủ bằng Ctrl+C. Nếu cổng đang dùng, chạy `python3 BiLSTM/scripts/pilot_selector.py --port 8790` và mở địa chỉ được in ra. Công cụ dùng dữ liệu trên máy chạy Python, không cần server bên ngoài hoặc thư viện frontend từ CDN.

## Chọn clip theo quy trình nào?

1. **Xem đề xuất ban đầu:** mặc định 40 clip, seed 42; có thể đặt mục tiêu 20–50 clip. Tạo đề xuất mới có bước xem trước rồi mới thay danh sách hiện tại; các clip đã loại tiếp tục được loại.
2. **Kiểm tra độ phủ:** bảng bên trái cho biết số clip theo hoạt động và hai nhóm. Lọc theo mã trẻ, phiên, tầng thiếu khớp, nhóm hoặc tên clip. Có tab tất cả, đã chọn, chờ xem và đã loại.
3. **Review trực quan:** xem mọi ứng viên skeleton và track được chọn, phát/dừng/kéo frame; đi tới cảnh báo track vắng, khuyết frame nguồn hoặc bước tọa độ lớn. Có bảng X/Y/confidence từng joint và viewer 2D/3D được sinh lại bằng cùng cấu hình Tracklet V2. Nếu thiếu ROMP, người dùng vẫn xem được OpenPose 2D; không suy ra độ chính xác hình học 3D từ bề mặt minh họa.
4. **Điều chỉnh có lý do:** chọn/bỏ clip, sửa lý do; loại clip cần ghi lý do. Khi đổi nhóm một clip, công cụ chuyển các clip khác của cùng mã trẻ sang cùng nhóm và báo số clip bị ảnh hưởng để giữ tách nhóm. Đánh dấu đã xem là thao tác của người review, không được tự suy từ việc tải/phát trang.
5. **Chốt:** cần đủ số clip mục tiêu, có clip trong cả hai nhóm, không trùng mã trẻ, có lý do chọn/loại và mọi clip chọn đã được đánh dấu xem. Thiếu độ phủ hoạt động/tầng được cảnh báo, không bị che giấu bằng cách lấy clip trùng hoặc tự thêm nhãn review.

Thuật toán đề xuất tách các mã trẻ trước, tìm partition đủ khả năng phủ lớp, phân bổ quota hoạt động trong từng nhóm, rồi đa dạng tầng thiếu khớp và mã trẻ/phiên. Dùng seed cố định và ID sắp xếp để tái tạo. Không dùng accuracy, đúng/sai của classifier hoặc nhãn `_y/_i/_n` để chọn. Đây là mẫu thăm dò có phân tầng, không phải mẫu để ước lượng tỷ lệ lỗi toàn quần thể.

## Kết quả khảo sát sơ bộ và bản nháp

Đã đo **1.374 clip / 254.674 frame**, thuộc **10 hoạt động / 32 mã trẻ**; không có clip bị cách ly trong lần chạy này. Số liệu tổng hợp có trong [pilot_selector_summary.json](../../BiLSTM/artifacts/pilot_selector_summary.json).

| Tầng theo tỷ lệ joint không hợp lệ | Toàn catalogue | Đề xuất 40 clip |
|---|---:|---:|
| Ít thiếu: ≤ 5% | 605 | 20 |
| Vừa: > 5% và ≤ 20% | 602 | 13 |
| Nhiều: > 20% | 167 | 7 |

Đề xuất seed 42 chia **20 tuning / 20 evaluation**, phủ đủ 10 hoạt động ở mỗi nhóm, mỗi hoạt động 2 clip/nhóm. Hai nhóm có 13 và 15 mã trẻ được chọn, không trùng nhau. **0/40 clip đã được người dùng review; chưa có pilot được chốt.** Đây là đề xuất có thể điều chỉnh, không phải quyết định nghiên cứu cuối cùng.

Các metric đo trên selected raw skeleton, với confidence threshold 0,1 theo config:

- `missing_joint_ratio`: joint không hợp lệ / số frame nguồn khả dụng × 25; khi target vắng thì joint slot là missing. Điều kiện hợp lệ kế thừa preprocessing, không phải nhãn pose đúng.
- Confidence trung bình chỉ dùng confidence hữu hạn của người được chọn; không gán confidence giả cho frame target vắng. Tỷ lệ confidence thấp dùng mẫu số các confidence hữu hạn và có tính confidence bằng 0 từ nguồn.
- Khoảng thiếu dài nhất tính cả mép clip, nhưng không nối qua file frame nguồn bị khuyết. Số file nguồn khuyết được ghi riêng, không giả là frame đã quan sát.
- Bước tọa độ lớn tính trên joint hợp lệ ở hai frame nguồn liên tiếp, ngưỡng 0,35 lần median bbox diagonal của track. Có thể là chuyển động thật, không phải verified jitter hay identity switch.
- Không gán FPS nguồn; tốc độ phát trên viewer chỉ phục vụ xem. Grouping theo subject ID của dataset không xác minh người được theo dõi thật sự là trẻ.

## Lưu, nhập và xuất

Bản nháp được tự lưu trên ổ đĩa máy chạy công cụ, không chỉ nằm trong browser storage. Có nút xuất/nhập JSON để sao lưu. Bản nháp sai catalogue hoặc cửa sổ có revision cũ bị từ chối thay vì ghi đè lựa chọn mới.

```text
BiLSTM/artifacts/pilot_selector/
  catalog.json                        catalogue hiện tại
  catalogues/<hash>.json               các phiên bản catalogue
  selections/<catalog-hash>/draft.json bản nháp của từng catalogue
  selections/<catalog-hash>/exports/<id>/
    manifest.csv                      toàn bộ clip chọn
    tuning.csv                        nhóm chỉnh tham số
    evaluation.csv                    nhóm đánh giá
    selection.json                    lý do, review, config, hash và protocol
```

Khi chốt, server kiểm tra lại code/config và nội dung frame nguồn của các clip được chọn. File xuất ghi source hash, catalogue/config hash, nhóm, metadata, metric và lý do. Mỗi lần chốt tạo thư mục phiên bản mới; không sửa các lần xuất trước. Trạng thái “đã xem” chỉ ghi việc review trực quan, không được diễn giải thành ground truth danh tính/tư thế.

Catalogue và lựa chọn cục bộ được bỏ qua bởi Git. Code, kiểm thử, tài liệu và bản tổng hợp khảo sát được quản lý trên nhánh nghiên cứu. Bản clone mới cần chuẩn bị MMASD rồi chạy tạo catalogue; không có clip giả để lấp vào chỗ thiếu dữ liệu.

## Mã nguồn và kiểm chứng

- [pilot_catalog.py](../../BiLSTM/scripts/pilot_catalog.py): đo raw, source hash, phát hiện thay đổi file trong lúc đọc và lưu catalogue có checksum.
- [pilot_sampling.py](../../BiLSTM/scripts/pilot_sampling.py): đề xuất có seed, quota, tách mã trẻ và validation bản nháp.
- [pilot_selector.py](../../BiLSTM/scripts/pilot_selector.py): server cục bộ, preview, lưu revision và xuất manifest.
- [Giao diện](../../apps/pilot_selector/index.html), [kiểm thử](../../BiLSTM/tests/test_pilot_selector.py).

Kiểm thử tự động không cần dữ liệu riêng: `make test-pilot`. Các trường hợp gồm tính tái tạo, số clip lẻ/chẵn, không trùng mã trẻ, shortage/exclusions, JSON sai cấu trúc/phiên bản, metric và gap, khôi phục bản nháp, chống ghi đè revision, CSV multiline và kiểm tra lại nguồn trước khi xuất.

Bộ kiểm thử MMASD + DRYAD hiện có **125 kiểm thử đạt**, trong đó nhóm pilot có **28 kiểm thử**. Các kiểm tra Python lint/format và cú pháp JavaScript cũng đạt.

Đã thử trên Chrome với dữ liệu thật: lọc, preview 2D/3D, autosave và reload, xuất/nhập draft, từ chối catalogue sai, giữ exclusions khi đề xuất lại, chốt và tải manifest 40 clip không trùng mã trẻ. Đã kiểm tra layout desktop/mobile, không có lỗi JavaScript. Review và export trong thử nghiệm trình duyệt được lưu riêng ở `/tmp`; không làm các clip trong bản nháp người dùng thành “đã xem”.

Chưa triển khai nhãn lỗi ground truth, hiệu chỉnh skeleton, benchmark corruption hoặc đánh giá detector. Bước tiếp theo là người dùng review/chốt pilot, rồi lập thống kê và đánh giá phương pháp trên tập đó.
