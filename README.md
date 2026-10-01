# NCKH Trẻ tự kỷ – Skeleton-based Action Recognition

Repo gồm hai nhánh thực nghiệm độc lập: nhận dạng hoạt động trị liệu trên skeleton MMASD bằng BiLSTM và phân tích gait/skeleton DRYAD bằng MSG3D. DREAM được dùng để nghiên cứu dữ liệu trị liệu và độ tin cậy của skeleton. Các bộ dữ liệu có nhãn và mục tiêu khác nhau; không gộp chúng thành một tác vụ phân loại.

## Bắt đầu từ đâu

| Nhu cầu | Vị trí |
|---|---|
| Chỉ mục tài liệu | [docs/README.md](docs/README.md) |
| Kế hoạch hướng nghiên cứu hiện tại | [docs/plans/](docs/plans/) |
| Pipeline MMASD, config, kiểm thử và hướng dẫn chạy | [BiLSTM/README.md](BiLSTM/README.md) |
| Script chuẩn bị/audit DRYAD | [tools/](tools/) |
| Mã MSG3D dùng cho DRYAD | [autism/](autism/) |
| Kiểm thử DRYAD | [tests/](tests/) |
| Báo cáo thực nghiệm MMASD | [BAO_CAO_THUC_NGHIEM_BILSTM.md](docs/reports/BAO_CAO_THUC_NGHIEM_BILSTM.md) |
| Đối chiếu ba dataset | [BAO_CAO_SO_SANH_MMASD_DRYAD_DREAM.md](docs/reports/BAO_CAO_SO_SANH_MMASD_DRYAD_DREAM.md) |
| Mapping DREAM | [BA_DAU_RA_PIPELINE_DREAM_MAPPING.md](docs/reports/BA_DAU_RA_PIPELINE_DREAM_MAPPING.md) |
| Contract DRYAD | [artifacts/dryad_msg3d/DATA_CONTRACT_v1.md](artifacts/dryad_msg3d/DATA_CONTRACT_v1.md) |

## Cấu trúc repo

```text
BiLSTM/             MMASD: src/, scripts/, configs/, tests/, artifacts/
tools/              Script DRYAD, kiểm tra cấu trúc repo
autism/             Mã MSG3D và cấu hình của nhánh gait
tests/              Kiểm thử contract và preprocessing DRYAD
docs/plans/         Kế hoạch nghiên cứu
docs/reports/       Báo cáo và tài liệu nghiên cứu
artifacts/          Manifest, báo cáo và output DRYAD
data_MMASD/         Dữ liệu MMASD cục bộ
Data_DRYAD/         Dữ liệu DRYAD cục bộ
data_Dream/         Dữ liệu DREAM cục bộ
```

Các thư mục dữ liệu cục bộ, checkpoint và những output lớn đã chọn được loại khỏi Git bằng [.gitignore](.gitignore). Không thay đổi dữ liệu gốc khi chạy script; manifest, config và artifact kết quả được giữ cạnh pipeline tương ứng. Các file kế hoạch mới đặt trong `docs/plans/` và thêm vào chỉ mục của thư mục đó.

Repo mẫu MMASD được khai báo là Git submodule; khi clone mới, dùng `git clone --recurse-submodules <repo-url>` hoặc chạy `git submodule update --init` sau khi clone. Thư mục `autism/` được đưa vào repo chính từ [saniazahan/autism](https://github.com/saniazahan/autism), commit `15048b39`, cùng các chỉnh sửa MSG3D cục bộ. Metadata Git cũ của nó được giữ tại `autism/.git.backup/` trên máy này và không đưa vào commit. File checkpoint `.pt` là tài nguyên cục bộ bị loại khỏi Git.

## Lệnh chung

Chạy từ thư mục gốc repo:

```bash
make help                 # xem lệnh có sẵn
make check                # kiểm tra cú pháp Python, link README/tài liệu; không cần thư viện ngoài
make test-bilstm          # kiểm thử MMASD; cần môi trường BiLSTM và pytest
make test-dryad           # kiểm thử DRYAD; cần môi trường DRYAD, dữ liệu/artifact cục bộ
make test                 # chạy cả hai nhóm kiểm thử
```

Đổi Python interpreter bằng `make PYTHON=/duong/dan/toi/python check` nếu dùng virtual environment hoặc conda. Dependencies MMASD nằm trong [BiLSTM/requirements.txt](BiLSTM/requirements.txt); nhánh DRYAD/MSG3D có môi trường riêng tại [autism/environment_autism.yml](autism/environment_autism.yml). `make check` chỉ dùng Python standard library.

Ví dụ lệnh riêng theo pipeline:

```bash
python3 BiLSTM/scripts/audit_paper_protocol.py --config BiLSTM/configs/paper_repro.yaml
python3 tools/read_dryad_data.py
```

Lệnh audit MMASD đọc manifest/split hiện có và ghi `BiLSTM/artifacts/paper_repro/protocol_audit.json`. Script DRYAD đọc workbook cục bộ và ghi hình vào `dryad_visualizations/`. Xem README của từng nhánh trước khi chạy training hoặc tái tạo artifact.
