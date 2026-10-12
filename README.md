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

## Chuẩn bị sau khi clone

```bash
git clone --recurse-submodules <repo-url>
cd nckh_tretuky
make check
```

`make check` kiểm tra cú pháp Python và các liên kết tài liệu thuộc repo bằng Python standard library; không cần tải dữ liệu. Nếu muốn chạy kiểm thử, cài dependencies của pipeline tương ứng và `pytest`, sau đó dùng `make test-bilstm` hoặc `make test-dryad`. Dữ liệu MMASD, DRYAD, DREAM và checkpoint cần được lấy riêng theo nguồn và quyền sử dụng của từng bộ dữ liệu, rồi đặt vào các thư mục cục bộ ghi trong bảng cấu trúc. Chúng không đi kèm bản clone. Một số báo cáo có liên kết tới đường dẫn dữ liệu/checkpoint cục bộ và chỉ mở được khi đã chuẩn bị các file đó.

## Chuẩn bị đưa lên GitHub

Chạy `make check` và `git status --short` trước khi commit. `.gitignore` loại dữ liệu thô, checkpoint, môi trường Python và các hình/HTML sinh tại `artifacts/visualizations/`; các script trong `tools/` và tài liệu nguồn vẫn cần được commit nếu muốn người khác tái tạo hình. Kiểm tra riêng các artifact đã được Git theo dõi trong `BiLSTM/artifacts/` trước khi công khai, vì `.gitignore` không loại những file đã được theo dõi từ trước. Repo dùng giấy phép [MIT](LICENSE) cho phần mã nguồn của dự án; dữ liệu và mã nguồn bên thứ ba có điều khoản riêng tại nguồn tương ứng.

## Lệnh chung

Chạy từ thư mục gốc repo:

```bash
make help                 # xem lệnh có sẵn
make check                # kiểm tra cú pháp Python, link README/tài liệu; không cần thư viện ngoài
make test-bilstm          # kiểm thử MMASD; cần môi trường BiLSTM và pytest
make test-dryad           # kiểm thử DRYAD; cần môi trường DRYAD, dữ liệu/artifact cục bộ
make test                 # chạy cả hai nhóm kiểm thử
```

Đổi Python interpreter bằng `make PYTHON=/duong/dan/toi/python check` nếu dùng virtual environment hoặc conda. Dependencies chung của dự án được khai báo tại [requirements.txt](requirements.txt) và [pyproject.toml](pyproject.toml); riêng MMASD có [BiLSTM/requirements.txt](BiLSTM/requirements.txt); nhánh DRYAD/MSG3D có môi trường Conda tại [autism/environment_autism.yml](autism/environment_autism.yml). `make check` chỉ dùng Python standard library.

Ví dụ lệnh riêng theo pipeline:

```bash
python3 BiLSTM/scripts/audit_paper_protocol.py --config BiLSTM/configs/paper_repro.yaml
python3 tools/read_dryad_data.py
```

Lệnh audit MMASD đọc manifest/split hiện có và ghi `BiLSTM/artifacts/paper_repro/protocol_audit.json`. Script DRYAD đọc workbook cục bộ và ghi hình vào `dryad_visualizations/`. Xem README của từng nhánh trước khi chạy training hoặc tái tạo artifact.

### Xem dữ liệu DREAM

```bash
python3 tools/visualize_dream.py
```

Lệnh này tạo `artifacts/visualizations/dream_viewer.html` từ một record JA mẫu. Mở file HTML bằng trình duyệt để phát lại skeleton thân trên ở hai góc nhìn, kéo đến từng frame, lọc khớp theo confidence và xem các thành phần `rx/ry/rz` của eye/head gaze. Trang HTML tự chứa dữ liệu của record được chọn, không cần máy chủ hay thư viện Python ngoài standard library.

Để xem một record khác hoặc một đoạn ngắn:

```bash
python3 tools/visualize_dream.py "data_Dream/DREAMdataset/User 33/User 33_10_intervention 1_20170623_090815.158000.json" --start-frame 500 --frames 1000 --output artifacts/visualizations/dream_imitation.html
```

Mỗi JSON là một đoạn dữ liệu, không phải một trial đã gán nhãn đúng/sai. Mũi tên gaze chỉ hướng vector; không suy ra vật thể được nhìn vì bản public thiếu tọa độ target và mốc stimulus. Một số record có metadata thời gian sai, nên viewer dùng độ dài mảng tín hiệu và `frame_rate` để xác định thời gian hiển thị.
