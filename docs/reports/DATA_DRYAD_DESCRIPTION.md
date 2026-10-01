# Mô tả bộ dữ liệu Dryad

## 1. Thông tin tổng quan

| Trường | Nội dung |
|---|---|
| Tên bộ dữ liệu | *Three dimensional dataset combining gait and full body movement of children with autism spectrum disorders collected by Kinect v2 camera* |
| DOI | [10.5061/dryad.s7h44j150](https://doi.org/10.5061/dryad.s7h44j150) |
| Tác giả | Ahmed A. Al-Jubouri, Israa Hadi Ali và Yasen Rajihy |
| Đơn vị | College of Information Technology, University of Babylon, Iraq |
| Ngày công bố | 11/09/2020 |
| Lĩnh vực | Khoa học máy tính và tâm lý học |
| Bài báo liên quan | [Generating 3D Dataset of Gait and Full Body Movement of Children with Autism Spectrum Disorders Collected by Kinect V2 Camera](https://ijact.in/index.php/ijact/article/view/1193) |
| Bài toán chính | Phân loại trẻ phát triển điển hình và trẻ có rối loạn phổ tự kỷ (ASD) dựa trên chuyển động |

Đây là bộ dữ liệu chuyển động 3D của trẻ em, tập trung vào dáng đi và chuyển động toàn thân. Dữ liệu được thu bằng cảm biến Microsoft Kinect v2 trong môi trường được kiểm soát. Mục tiêu là cung cấp các đặc trưng vận động có thể sử dụng cho phân tích dáng đi, nhận dạng mẫu chuyển động và xây dựng mô hình học máy hỗ trợ nghiên cứu ASD.

> Bộ dữ liệu chỉ phục vụ nghiên cứu và phát triển mô hình. Nó không thay thế chẩn đoán hoặc đánh giá lâm sàng của chuyên gia.

## 2. Bộ dữ liệu giải quyết vấn đề gì?

Việc đánh giá chuyển động và dáng đi của trẻ có ASD thường cần quan sát trực tiếp, phụ thuộc vào chuyên gia và có thể khó chuẩn hóa giữa các lần đánh giá. Bộ dữ liệu này tạo ra một nguồn dữ liệu định lượng để nghiên cứu các khác biệt về:

- quỹ đạo và tọa độ 3D của các khớp cơ thể;
- góc giữa các khớp;
- khoảng cách giữa các bộ phận cơ thể và khoảng cách tới mặt đất;
- các thông số của chu kỳ dáng đi như thời gian, vận tốc, chiều dài và chiều rộng bước chân;
- chuyển động toàn thân và các đặc trưng liên quan đến tay, chân, đầu gối, khuỷu tay và cổ tay.

Nhờ đó, dữ liệu có thể hỗ trợ các bài toán như:

1. phân loại ASD và phát triển điển hình;
2. phân tích đặc điểm vận động và dáng đi;
3. đánh giá các đặc trưng chuyển động có khả năng phân biệt hai nhóm;
4. thử nghiệm các thuật toán học máy, học sâu và giải thích mô hình;
5. so sánh các phương pháp biểu diễn dữ liệu skeleton và dữ liệu dáng đi.

## 3. Đối tượng và quy trình thu thập

Theo `readme.docx`, dữ liệu ban đầu gồm:

- 50 trẻ phát triển điển hình;
- 50 trẻ có ASD;
- một nhóm gồm 9 trẻ có mức độ ASD nặng được mô tả riêng trong metadata;
- dữ liệu được thu trong điều kiện có kiểm soát về nhiệt độ, ánh sáng và vị trí camera.

Quy trình thu thập chính:

1. Trẻ đi bộ theo một đường thẳng, với tốc độ bình thường, hướng về phía Kinect v2.
2. Mỗi lần đi dài khoảng 1,5–4 m và bao gồm khoảng hai chu kỳ dáng đi.
3. Camera ghi nhận nhiều lần; một chu kỳ dáng đi phù hợp được chọn để xử lý.
4. Kinect được đặt ở độ cao khoảng 0,75 m và bật trước khoảng 30 phút để giảm ảnh hưởng của hiện tượng trôi cảm biến.
5. Điều kiện môi trường được ghi nhận trong khoảng 20–22°C và 76–87 Lux theo tài liệu mô tả.

Nguồn dữ liệu gốc bao gồm skeleton 3D, video skeleton, video quỹ đạo khớp và video màu. Phiên bản đang có trong thư mục dự án gồm file Excel đã tổng hợp cùng với file nén `Dataset-2.rar`.

## 4. Cấu trúc thư mục cục bộ

```text
Data_DRYAD/
└── doi_10_5061_dryad_s7h44j150__v20200911/
    ├── Dataset-2.rar
    └── Dataset-2/
        └── Dataset/
            ├── Final dataset.xlsx
            ├── Informed Consent Form for parents of children.jpg
            └── readme.docx
```

File được sử dụng trực tiếp cho phân tích bảng là:

```text
Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/Final dataset.xlsx
```

## 5. Cấu trúc file `Final dataset.xlsx`

File có một sheet tên `Sheet1`. Sau khi bỏ phần cột trống do định dạng Excel ở cuối bảng, dữ liệu thực tế gồm:

| Thành phần | Số lượng / nội dung |
|---|---|
| Số mẫu | 800 |
| Số thuộc tính đầu vào | 1.259 |
| Cột nhãn | 1 cột `class` |
| Tổng số cột có dữ liệu | 1.260 |
| Nhãn `A` | 400 mẫu |
| Nhãn `T` | 400 mẫu |
| Giá trị thiếu | README ghi nhận không có giá trị thiếu |

Trong đó, `class` là biến mục tiêu. Theo ngữ cảnh của bộ dữ liệu, `A` tương ứng với nhóm ASD và `T` tương ứng với nhóm phát triển điển hình.

### Nhóm đặc trưng chính

Các cột trong file gồm những nhóm sau:

| Nhóm | Mô tả | Ví dụ / đơn vị |
|---|---|---|
| `mean-*` | Giá trị trung bình của tọa độ, góc hoặc khoảng cách | mét, độ |
| `variance-*` | Phương sai của đặc trưng | m² hoặc deg² |
| `std-*` | Độ lệch chuẩn của đặc trưng | mét hoặc độ |
| `Rom*` | Phạm vi chuyển động của khớp theo trục x/y | mét |
| Thông số dáng đi | Chiều dài bước, chiều rộng bước, chu kỳ dáng đi, thời gian stance/swing, vận tốc | m, ms, m/s |
| Đặc trưng nhị phân | Vị trí đầu ngón tay trái/phải so với Spine Base | 0 hoặc 1 |
| `MaxDBFE`, `MinDBFE`, `Threshold` | Khoảng cách lớn nhất/nhỏ nhất giữa hai bàn chân và ngưỡng | mét |
| `class` | Nhãn phân loại | `A` hoặc `T` |

### Khớp và góc

Hệ thống skeleton sử dụng 25 khớp cơ thể, gồm đầu, cổ, vai, khuỷu tay, cổ tay, bàn tay, hông, đầu gối, mắt cá chân và bàn chân. Ngoài tọa độ khớp, dữ liệu còn chứa 16 loại góc giữa các khớp, chẳng hạn:

- góc vai trái/phải;
- góc khuỷu tay trái/phải;
- góc cổ tay trái/phải;
- góc hông, đầu gối và mắt cá chân trái/phải.

## 6. Tiền xử lý và sử dụng cho học máy

Một quy trình phân tích phù hợp có thể gồm:

1. Đọc `Final dataset.xlsx` bằng `pandas`.
2. Loại bỏ các cột không có tên hoặc hoàn toàn rỗng do định dạng Excel.
3. Tách `class` làm nhãn mục tiêu.
4. Kiểm tra kiểu dữ liệu, giá trị thiếu và các giá trị bất thường.
5. Chuẩn hóa các đặc trưng số vì các nhóm có đơn vị và thang đo khác nhau.
6. Chia dữ liệu thành train/validation/test.
7. Huấn luyện mô hình phân loại như Random Forest, SVM, XGBoost hoặc mạng nơ-ron.
8. Đánh giá bằng accuracy, precision, recall, F1-score và ma trận nhầm lẫn.

### Lưu ý quan trọng khi chia dữ liệu

Không nên chia ngẫu nhiên từng hàng nếu nhiều hàng có nguồn gốc từ cùng một trẻ hoặc cùng một mẫu sau augmentation. Cách chia này có thể làm dữ liệu gần như giống nhau xuất hiện ở cả tập huấn luyện và tập kiểm tra, dẫn đến data leakage và kết quả quá lạc quan. Nên chia theo participant/session trước khi đánh giá mô hình nếu metadata cho phép.

## 7. Lưu ý về augmentation và tính nhất quán tài liệu

Tài liệu mô tả cho biết dữ liệu được augmentation bằng bảy phép biến đổi. README cũng ghi 400 mẫu cho mỗi nhóm sau augmentation, tổng cộng 800 mẫu. Tuy nhiên, phần abstract trên Dryad lại mô tả 700 thư mục, gồm 350 thư mục cho mỗi nhóm. Vì vậy, khi viết báo cáo hoặc công bố kết quả, cần:

- ghi rõ đang sử dụng file tổng hợp `Final dataset.xlsx` với 800 mẫu;
- phân biệt số trẻ ban đầu với số mẫu sau augmentation;
- không xem các mẫu augmentation là những người tham gia độc lập;
- đối chiếu lại phiên bản dữ liệu và tài liệu gốc nếu cần tái lập hoàn toàn thí nghiệm.

Ngoài ra, phần header của file Excel có một số cột trống ở cuối do định dạng bảng. Khi đọc bằng phần mềm phân tích, chỉ nên giữ 1.260 cột có dữ liệu, trong đó 1.259 cột là thuộc tính và cột cuối `class` là nhãn.

## 8. Đạo đức và quyền riêng tư

Đây là dữ liệu liên quan đến trẻ em và có video màu của người tham gia. File cục bộ có `Informed Consent Form for parents of children.jpg`, cho thấy tài liệu đồng thuận của phụ huynh được cung cấp cùng bộ dữ liệu. Khi sử dụng dữ liệu:

- không cố gắng nhận dạng người tham gia;
- không phát tán lại video hoặc thông tin cá nhân ngoài phạm vi được phép;
- chỉ sử dụng cho mục đích nghiên cứu phù hợp;
- báo cáo rõ nguồn dữ liệu và DOI;
- không diễn giải kết quả mô hình như một công cụ chẩn đoán lâm sàng đã được kiểm định.

## 9. Trích dẫn đề xuất

Al-Jubouri, A. A., Hadi, I. H., & Rajihy, Y. (2020). *Three dimensional dataset combining gait and full body movement of children with autism spectrum disorders collected by Kinect v2 camera* [Dataset]. Dryad. https://doi.org/10.5061/dryad.s7h44j150

## 10. Nguồn tham khảo

- [Trang bộ dữ liệu trên Dryad](https://datadryad.org/dataset/doi%3A10.5061/dryad.s7h44j150)
- [DOI của bộ dữ liệu](https://doi.org/10.5061/dryad.s7h44j150)
- `Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/readme.docx`
- `Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/Final dataset.xlsx`
