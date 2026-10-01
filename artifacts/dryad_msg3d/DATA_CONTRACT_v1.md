# Round 2.5 — Dryad/MSG3D Data Contract v1

Ngày chốt: 2026-09-17  
Phạm vi: skeleton `.xlsx` trong `Dataset-2/Dataset`, không dùng `Final dataset.xlsx` làm đầu vào trực tiếp cho MSG3D.  
Machine-readable version key: `DATA_CONTRACT_v1`  
Trạng thái: **SAFE_TO_BUILD_PREPROCESSING**, với các giới hạn và mức độ tin cậy bên dưới.

Contract này mô tả một pipeline tái lập được cho bước dữ liệu. Nó chưa cho phép huấn luyện, chọn hyperparameter, tạo official folds hoặc đánh giá kết quả cuối cùng.

## 1. Input dataset

- Gốc: `Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/`.
- Input skeleton: các file `video/8.xlsx` và `augmentation/{1,2,3,4,5,6,7}/{same_id}.xlsx` theo metadata thư mục; công cụ Round 2 đã quét 800 file skeleton.
- Phân bố: 100 file/original subjects và 700 file/augmentation; ASD 400 file, TD 400 file.
- Nhãn thư mục: `Autism/children with ASD` -> `ASD` -> label số `1`; `Typical` -> `TD` -> label số `0`.
- File `Final dataset.xlsx` có 800 hàng, 1.259 thuộc tính và cột class (`A`/`T`). File này được dùng để kiểm tra provenance của feature, không phải nguồn thay thế cho chuỗi skeleton khi chạy MSG3D.
- Mỗi skeleton có 25 joint, 3 giá trị/joint và số frame quan sát `T_raw` từ 21 đến 60. Cột timestamp đầu tiên bị loại khỏi tensor.
- Đơn vị/toạ độ giữ nguyên theo workbook raw; **không scale theo subject** trong contract v1.

Mức tin cậy: `VERIFIED` cho số lượng file/shape raw sau khi quét; `HIGH_CONFIDENCE_RECONSTRUCTION` cho ánh xạ vào graph NTU.

## 2. Chosen M policy

- `M = 2`.
- Person slot 0: skeleton thực của sample.
- Person slot 1: vector zero cho toàn bộ `T=60`, shape `[3,60,25]`.
- Không dùng bản sao dữ liệu vào person slot 1.
- Không chọn `num_person=1`: checkpoint NTU120 có `data_bn` tương ứng `2 * 3 * 25 = 150` input features; đổi sang một person sẽ không còn là backbone tương thích exact-shape.

Decision: `ZERO_PAD_SECOND_PERSON`.  
Evidence: `VERIFIED` cho yêu cầu shape/checkpoint và `HIGH_CONFIDENCE_RECONSTRUCTION` cho dataset một người. Mã upstream MS-G3D khởi tạo tensor cố định với `max_body_true=2` và bố trí input cuối cùng theo `[3,T,25,2]`; xem [official NTU generator](https://github.com/kenziyuliu/MS-G3D/blob/master/data_gen/ntu_gendata.py).

## 3. Joint mapping

Raw header order:

```text
Midspain, AnkleLeft, AnkleRight, ElbowLeft, ElbowRight,
FootLeft, FootRight, HandLeft, HandRight, HandTipLeft,
HandTipRight, Head, HipLeft, HipRight, KneeLeft, KneeRight,
Neck, ShoulderLeft, ShoulderRight, SpineBase, SpineShoulder,
ThumbLeft, ThumbRight, WristLeft, WristRight
```

Model order đóng băng:

```text
SpineBase, SpineMid, Neck, Head, ShoulderLeft, ElbowLeft,
WristLeft, HandLeft, ShoulderRight, ElbowRight, WristRight,
HandRight, HipLeft, KneeLeft, AnkleLeft, FootLeft, HipRight,
KneeRight, AnkleRight, FootRight, SpineShoulder, HandTipLeft,
ThumbLeft, HandTipRight, ThumbRight
```

Permutation raw index -> model index (zero-based):

```python
RAW_TO_MODEL = (
    1, 14, 18, 5, 9, 15, 19, 7, 11, 21,
    23, 3, 12, 16, 13, 17, 2, 4, 8, 0,
    20, 22, 24, 6, 10,
)
```

Decision: `JOINT_MAPPING=FROZEN`.  
Evidence: `HIGH_CONFIDENCE_RECONSTRUCTION`, không phải `VERIFIED`: graph `ntu_rgb_d.py` công bố edge theo chỉ số nhưng không công bố semantic name; tên raw của Dryad giúp khôi phục chuỗi giải phẫu và mapping đã được kiểm tra là bijection.

Không reorder theo header của `Final dataset.xlsx`; tensor MSG3D luôn dùng model order ở trên.

## 4. Coordinate order

- `C = 3`.
- Contract v1: `COORD_ORDER = XYZ`.
- Raw workbook không có subheader `x/y/z`, nên không gắn nhãn VERIFIED chỉ từ header.
- Bằng chứng chéo: README mô tả 3D joint positions; feature `mean-x/y/z-*` trong Final dataset khớp với mean theo ba component raw; phân bố hình học cho thấy component 2 là depth hợp lý hơn hai component đầu.
- Không hoán đổi component, không đổi dấu, không scale và không centering/rotation bổ sung trong contract v1.

Status: `XYZ_HIGH_CONFIDENCE`, với giới hạn “raw axis labels absent”.

## 5. Temporal policy

- `TARGET_T = 60`.
- Nếu `T_raw < 60`: giữ toàn bộ frame hợp lệ ở đầu chuỗi, right-pad zero đến 60.
- Nếu `T_raw == 60`: giữ nguyên.
- Nếu `T_raw > 60`: fail-fast trong manifest builder; không tự truncate.
- Lưu kèm `valid_frame_count=T_raw` và `target_frame_count=60` để không nhầm frame pad với frame quan sát.

Decision: `RIGHT_ZERO_PAD_TO_T60`.  
Evidence: `HIGH_CONFIDENCE_RECONSTRUCTION` cho contract Dryad vì max quan sát là 60 và chính sách không tạo frame giả. Đây **không** phải tuyên bố rằng tác giả Dryad đã dùng đúng padding này. Upstream MS-G3D có bước chuẩn hoá riêng: tensor được cấp phát fixed-length và `pre_normalization` có thể lặp lại prefix để lấp phần đuôi null; xem [official preprocessing](https://github.com/kenziyuliu/MS-G3D/blob/master/data_gen/preprocess.py). Vì nguồn Dryad không ghi lại preprocessing đó, contract v1 chọn chính sách bảo toàn dữ liệu raw và ghi rõ `valid_frame_count`.

Các lựa chọn `T=64`, repeat-last, linear interpolation và dynamic batch không được chọn trong v1: chúng thêm giả định hoặc làm mất tính so sánh shape cố định. Chẩn đoán forward-only trên sample `Typical/1/video/8.xlsx` (`T_raw=34`) cho thấy native, zero-pad và repeat-prefix đều hữu hạn; logits không được dùng để chọn chính sách.

## 6. Sentinel policy

- Giá trị phát hiện: `100.0` đúng bằng floating-point equality trên 75 coordinate columns đầu.
- Quan sát: 101 occurrences, 58 cặp `(file,frame)`, 3 ASD original subjects (`4, 8, 22`), 0 TD subject; longest run 12.
- `SENTINEL_POLICY = PRESERVE_RAW_WITH_FLAG`.
- Không sửa workbook raw, không clip, không interpolate, không thay 100 bằng NaN/0.
- Manifest cần có: `had_sentinel`, `sentinel_count`, `sentinel_joints`, `sentinel_components`, `repair_method=none`.
- Nhãn ngữ nghĩa của 100 chưa tìm thấy trong README/workbook: `LIKELY_SENTINEL`, semantics `UNKNOWN`.

Bằng chứng quyết định: với các sample affected, feature statistics tương ứng trong `Final dataset.xlsx` khớp với phép tính **bao gồm** 100 (sai số khoảng `1e-12` ở phần stats đã đối chiếu), trong khi loại 100 cho sai khác rất lớn. Vì vậy preserve là lựa chọn reproduction-like có bằng chứng mạnh nhất; nó không có nghĩa 100 là measurement sinh lý hợp lệ.

Evidence cho cách xử lý: `HIGH_CONFIDENCE_RECONSTRUCTION`; evidence cho ý nghĩa vật lý: `UNKNOWN`.

## 7. Duplicate graph

Nguồn CSV:

- [`duplicate_groups.csv`](duplicate_groups.csv): 44 numeric duplicate groups, 117 file rows.
- [`duplicate_subject_components.csv`](duplicate_subject_components.csv): 88 connected components trên 100 subjects; 7 component có nhiều subject, 81 singleton, component lớn nhất có 5 subject.

Phát hiện bổ sung: có 2 byte-identical groups, đều trong ASD augmentation; numeric duplicate graph rộng hơn byte hash vì có workbook khác byte nhưng cùng parsed numeric cells.

## 8. Split protocol

Hai protocol được định nghĩa, không tạo fold ở Round 2.5:

1. `LEAKAGE_SAFE_COMPONENT_SPLIT` — protocol chính. Xem mỗi connected component trong duplicate graph là một split unit; mọi subject trong component phải cùng một partition. Không để cùng `numeric_hash` ở train và test.
2. `REPRODUCTION_LIKE_SUBJECT_SPLIT` — protocol phụ. Chia theo subject folder nhưng cho phép các duplicate-connected subjects ở khác partition; báo cáo riêng vì có nguy cơ leakage.

Không chia ngẫu nhiên từng file/augmentation. Không coi 800 file là 800 người. Subject IDs chỉ là provenance trong dataset release, không phải danh tính cá nhân.

## 9. Provenance policy

Mỗi sample manifest tương lai phải lưu tối thiểu:

```text
class, subject_id, source, augmentation_id, source_path,
raw_sha256, numeric_hash, valid_frame_count, target_frame_count,
m_policy, joint_mapping_version, coord_order,
had_sentinel, sentinel_count, repair_method, duplicate_component_id
```

Raw `.xlsx` là immutable input. Các CSV/MD trong `artifacts/dryad_msg3d/` chỉ là diagnostic provenance; không thay thế raw và không phải tensor đầu ra.

## 10. Model input contract

Tensor trước MSG3D:

```text
[N, C, T, V, M] = [N, 3, 60, 25, 2]
dtype = float32
```

Thứ tự là `C=XYZ`, `V=MODEL_JOINTS`, `M=(real_person, zero_person)`. `data_bn` phải có `150` features. Backbone plain MSG3D dùng `dream=False`, `use_angle=False`, 2-class FC head; không đưa `Final dataset.xlsx` feature vector 1.259 chiều vào model này.

## 11. Out-of-contract behavior

- Missing/non-numeric coordinate trong 75 coordinate columns: fail-fast, không tự điền.
- `T_raw > 60`: fail-fast.
- Subject/class không parse được: fail-fast.
- Duplicate graph không dùng để xoá sample; chỉ dùng để kiểm soát split và provenance.
- Sentinel không bị xoá/repair trong v1.
- Không chạy training/evaluation trong bước contract verification.

## 12. Verification checklist

- [x] 800 raw skeleton files đã quét; ASD/TD cân bằng theo file release.
- [x] Mapping là bijection 25 joint; test tĩnh đã pass.
- [x] `M=2` và `data_bn=150` được kiểm tra bằng static forward/checkpoint load.
- [x] Native sample, `T=60` zero-pad và các temporal diagnostic forward cho output hữu hạn.
- [x] Exact `100` được đếm và đối chiếu với feature stats Final dataset.
- [x] Duplicate groups và subject connected components đã xuất thành CSV.
- [ ] Chưa tạo manifest `.npy/.pkl`, official folds hoặc train/validation/test split.
- [ ] Chưa chứng minh mapping/XYZ/temporal policy là lựa chọn nguyên bản của tác giả Dryad.

## 13. Decision gate

**SAFE_TO_BUILD_PREPROCESSING = YES**, với các điều kiện:

- chỉ build manifest/tensor theo đúng contract v1;
- không sửa raw và không âm thầm đổi policy;
- primary report dùng `LEAKAGE_SAFE_COMPONENT_SPLIT`, còn protocol reproduction-like phải được gắn nhãn riêng;
- mọi run phải ghi contract version, git diff và provenance fields;
- chưa được xem kết quả model là kết quả tái lập paper cho đến khi hoàn thành sensitivity checks về sentinel, normalization và temporal policy.

Round 3 được phép bắt đầu ở mức **data-manifest/preprocessing dry-run**. Việc huấn luyện chỉ nên mở sau khi dry-run kiểm tra shape, hash, label count, sentinel flag và component isolation đều pass.
