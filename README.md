# NCKH Trẻ tự kỷ – Chất lượng dữ liệu skeleton và phân tích hành vi

Hướng nghiên cứu hiện tại là **đánh giá, phát hiện và hiệu chỉnh lỗi dữ liệu khung xương của trẻ em**, kiểm chứng chủ yếu trên dữ liệu trẻ tự kỷ (ASD). Trước mắt dùng skeleton MMASD có sẵn; đánh giá khả năng phục hồi và bảo toàn chuyển động là trọng tâm, nhận dạng hành động là phép kiểm chứng bổ sung.

Repo đã có hai nhánh thực nghiệm làm nền tảng: nhận dạng hoạt động trị liệu trên MMASD bằng BiLSTM và phân tích gait/skeleton DRYAD bằng MSG3D. DREAM được dùng để khảo sát dữ liệu trị liệu và độ tin cậy của skeleton. Các bộ dữ liệu có nhãn và mục tiêu khác nhau; không gộp chúng thành một tác vụ phân loại.

## Tiến độ hiện tại

**Cập nhật ngày 18/10/2026:** đã có pipeline nhận dạng MMASD, phân tích chất lượng đầu vào ban đầu và viewer review 2D/3D cho 120 clip. Mô hình đơn MMASD đạt accuracy 86,22%; stacking đạt 89,05% trên 283 clip **validation**. Pipeline phát hiện–hiệu chỉnh theo hướng nghiên cứu mới đang ở giai đoạn chuẩn bị, chưa có kết quả benchmark trước/sau.

**Đã hoàn thành bước 1 — rà soát và khóa pipeline quan sát:** cố định OpenPose BODY_25 2D + Tracklet V2, đo trước tiền xử lý; kiểm tra trên 3 clip thật / 470 frame và 22 kiểm thử. Có [báo cáo pipeline và đối chiếu số liệu](docs/reports/RA_SOAT_VA_KHOA_PIPELINE_MMASD.md), cấu hình độc lập và snapshot hash. Bước tiếp theo là chọn pilot 20–50 clip; split cũ chỉ tách phiên, vẫn có 13 subject ID chung giữa train/validation.

**Đã chuẩn hóa code và đẩy lên GitHub:** tiếp tục nghiên cứu trên nhánh [research/skeleton-reliability](https://github.com/dekii2275/nckh_tretuky/tree/research/skeleton-reliability), commit chuẩn hóa [c2758ab](https://github.com/dekii2275/nckh_tretuky/commit/c2758ab6bfddf845fd125a9842ac950abbae6d2b). Script audit đã được chia thành các hàm đọc dữ liệu, đối chiếu thực nghiệm, smoke test và kiểm tra snapshot; mặc định dùng [cấu hình v0.2](BiLSTM/configs/skeleton_audit_v0_2.yaml). Snapshot v0.1 được giữ làm lịch sử.

| Kiểm chứng bản code chuẩn hóa | Kết quả |
|---|---|
| Bộ kiểm thử MMASD + DRYAD trên máy | **97 kiểm thử đạt** |
| Kiểm thử audit trên checkout sạch, không cần dữ liệu riêng | **13 kiểm thử đạt** |
| Audit dữ liệu thật và bản khóa v0.2 | **3 clip / 470 frame; verify 545 file đạt** |
| Lint/format và kiểm tra cú pháp/liên kết tài liệu | **Đạt** |
| GitHub Actions của commit `c2758ab` | **[Đạt](https://github.com/dekii2275/nckh_tretuky/actions/runs/39143719340)** |

**Đã có công cụ chọn pilot trực quan:** catalogue đã đo **1.374 clip / 254.674 frame**, 10 hoạt động và 32 mã trẻ với Tracklet V2 trước tiền xử lý. Đề xuất mặc định có 40 clip, chia 20 chỉnh tham số / 20 đánh giá, mỗi nhóm phủ 10 hoạt động và không trùng mã trẻ. Đây là **bản nháp, 0/40 clip được người dùng review**, chưa phải pilot đã chốt. Xem [công cụ và kết quả kiểm chứng](docs/reports/CONG_CU_CHON_PILOT_MMASD.md).

**Việc tiếp theo:** mở công cụ, xem và điều chỉnh danh sách, ghi lý do chọn/loại, đánh dấu đã xem rồi chốt manifest. Sau đó xuất thống kê chi tiết theo joint/frame và đánh giá hiệu chỉnh trên pilot. Xem [quy trình phát triển và kiểm tra](docs/DEVELOPMENT.md) để chạy lại audit hoặc tạo snapshot mới.

Xem [tiến độ nghiên cứu và nhật ký cập nhật](docs/reports/TIEN_DO_NGHIEN_CUU.md) để biết những việc đã làm, bằng chứng, hạn chế và bước tiếp theo. Sau mỗi đợt công việc có thay đổi về tiến độ hoặc kết quả, cập nhật trạng thái và thêm mục nhật ký trong trang này; đồng bộ phần tóm tắt README khi có mốc mới. Chỉ ghi hoàn thành khi đã có đầu ra và kiểm chứng tương ứng.

## Bắt đầu từ đâu

| Nhu cầu | Vị trí |
|---|---|
| Chỉ mục tài liệu | [docs/README.md](docs/README.md) |
| Đã làm được gì, kết quả và việc tiếp theo | [Tiến độ nghiên cứu](docs/reports/TIEN_DO_NGHIEN_CUU.md) |
| Nhánh nghiên cứu, chạy audit và kiểm tra | [Quy trình phát triển](docs/DEVELOPMENT.md) |
| Chọn pilot bằng giao diện và xuất manifest | [Công cụ chọn clip MMASD](docs/reports/CONG_CU_CHON_PILOT_MMASD.md) |
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
make test-audit           # kiểm thử raw observation và snapshot; không cần dữ liệu riêng
make audit-pipeline       # tạo snapshot v0.2 khi output chưa tồn tại; cần dữ liệu cục bộ
make verify-pipeline      # kiểm tra snapshot v0.2 và các input cục bộ đã khóa
make pilot-catalog        # đo sơ bộ toàn MMASD và tạo catalogue; cần dữ liệu cục bộ
make pilot-selector       # chạy giao diện chọn pilot tại localhost:8787
make test-pilot           # kiểm thử chọn mẫu/lưu bản nháp/xuất; không cần dataset
```

### Chọn clip bằng giao diện

Catalogue trên máy hiện tại đã được tạo. Chạy `make pilot-selector`, rồi mở `http://127.0.0.1:8787` bằng trình duyệt. Có thể lọc theo hoạt động, mã trẻ, phiên và mức thiếu khớp; xem skeleton 2D hoặc đối chiếu 2D/3D; sửa nhóm và lý do chọn/loại. Bản nháp tự lưu trên ổ đĩa. Chỉ chốt khi đủ số clip, mỗi clip có lý do/đã xem và hai nhóm không trùng mã trẻ; công cụ kiểm tra lại dữ liệu nguồn trước khi tạo manifest tổng, `tuning.csv`, `evaluation.csv` và hồ sơ JSON.

Sau khi clone hoặc cập nhật dữ liệu/code khảo sát, chạy `make pilot-catalog` trước. Catalogue, bản nháp, review và file xuất nằm trong `BiLSTM/artifacts/pilot_selector/`, được giữ cục bộ. Các chỉ số chỉ hỗ trợ chọn mẫu; không xem confidence, missingness hay bước tọa độ lớn là nhãn pose đúng/sai.

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
