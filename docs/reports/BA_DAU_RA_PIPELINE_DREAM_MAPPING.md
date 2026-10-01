# Ba đầu ra bổ sung: pipeline tạo dataset, tái dựng task DREAM và mapping liên bộ

Ngày đối chiếu: 01/10/2026  
Baseline: [`Ghi chép nội dung nghiên cứu hàng tuần.pdf`](../../Ghi%20chép%20nội%20dung%20nghiên%20cứu%20hàng%20tuần.pdf).  
Phạm vi kiểm tra: tài liệu gốc, schema/README và toàn bộ cấu trúc dữ liệu MMASD, DRYAD, DREAM đang có trong workspace. Báo cáo này không lặp lại các bảng mô tả participant/license/privacy/thiết bị của baseline, trừ thông tin cần cho pipeline hoặc để giải quyết mâu thuẫn.

## Quy ước bằng chứng

- **KBC**: không báo cáo/không có đủ bằng chứng trong các nguồn đã kiểm tra. KBC không có nghĩa là bước đó chắc chắn không xảy ra.
- **Protocol gốc**: điều tác giả mô tả đã làm khi thu thập.
- **Public artifact**: trường/file thực sự có trong bản dữ liệu hiện có.
- Không suy diễn hậu tố filename, giá trị bất thường, hay tín hiệu liên tục thành nhãn hành vi nếu không có codebook.

### Chỉ mục nguồn

| Mã | Nguồn |
|---|---|
| M0 | Baseline [`Ghi chép nội dung nghiên cứu hàng tuần.pdf`](../../Ghi%20chép%20nội%20dung%20nghiên%20cứu%20hàng%20tuần.pdf), nhất là mục 3, 6.1, 6.2, 7, 8 |
| M1 | Paper MMASD, mục 3.1–3.3 và 4.1–4.5: [arXiv 2306.08243](https://arxiv.org/pdf/2306.08243) |
| M2 | [`MMASD.../README.md`](../../MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis/README.md) |
| M3 | Dữ liệu MMASD: [`data_MMASD/`](../../data_MMASD/) và [`ADOS_rating.xlsx`](../../data_MMASD/ADOS_rating.xlsx) |
| D1 | README gốc DRYAD: [`readme.docx`](../../Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/readme.docx) |
| D2 | Cây dữ liệu DRYAD: [`Dataset/`](../../Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/) và [`Final dataset.xlsx`](../../Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/Final%20dataset.xlsx) |
| D3 | Audit đã kiểm thử trên toàn bộ 800 skeleton: [`DATA_CONTRACT_v1.md`](../../artifacts/dryad_msg3d/DATA_CONTRACT_v1.md) |
| R1 | Paper DREAM, mục 3, 3.1, 4 và Appendix A: [PLOS ONE e0236939](https://doi.org/10.1371/journal.pone.0236939) |
| R2 | [`DREAM-data.pdf`](../../data_Dream/documentation/DREAM-data.pdf) và [`dataset_tools/__init__.py`](../../data_Dream/documentation/dataset_tools/__init__.py) |
| R3 | Schema [`dream.1.2.json`](../../data_Dream/dream.1.2.json) và record mẫu [`data_example.json`](../../data_Dream/data_example.json) |
| R4 | Toàn bộ 3.121 record trong [`DREAMdataset/`](../../data_Dream/DREAMdataset/) |

---

# 1. Dataset creation pipeline

## 1.1 MMASD

`Participant → Setup → Task → Recording → Segmentation → Annotation → Preprocessing → Final sample`

| Bước | Pipeline tái dựng | Bằng chứng | KBC / thông tin còn thiếu |
|---|---|---|---|
| **Participant** | Tuyển trẻ ASD; screening bằng SCQ, eligibility bằng ADOS-2 và clinical judgment; Vineland đánh giá adaptive functioning. Metadata lâm sàng sau đó được nối ở cấp participant. | M1 §3.1; M3 `ADOS_rating.xlsx`. | Thiếu roster phát hành chính thức ánh xạ chính xác participant ↔ toàn bộ clip. Workbook có nhiều ID hơn roster 32 trẻ công bố nên không được dùng một mình để quyết định membership. |
| **Setup** | Can thiệp tại nhà; một RGB camera hướng vào trẻ. Tương tác tam giác child–trainer–adult model; dụng cụ phụ thuộc theme: robot, nhạc cụ, yoga. | M1 §3.2–3.3. | KBC camera calibration, vị trí/khoảng cách/góc cố định, đồng bộ âm thanh, và cách định danh child khi nhiều người cùng khung hình. |
| **Task** | Ba theme, 11 activity. Robot: trẻ theo/bắt chước vận động; Rhythm: trẻ và therapist chơi nhạc/hát; Yoga: therapist dẫn động tác. | M1 §4 và Table 3; M2. | Không có discrete-trial specification, stimulus script, số lần lặp, success criterion hay feedback/retry theo trial. Nghĩa hậu tố `_y/_i/_n` trong filename **KBC**; không được tự đổi thành correct/incorrect/no-response. |
| **Recording** | Lịch 10 tuần, session dự kiến 4 lần/tuần, khoảng 45 phút; toàn bộ training session được quay, tổng source >108 giờ. Welcoming/debrief tạo time labels hỗ trợ khoanh vùng đoạn cần xét. | M1 §3.2, §4. | KBC số session thực tế/participant, audio có được thu hay không, và provenance từ raw video tới từng clip public. |
| **Segmentation** | Người xử lý tìm thủ công timestamp bắt đầu/kết thúc của activity, cắt thành clip, xếp activity; bỏ clip <3 giây và dữ liệu nhiễu do video/ánh sáng/occlusion; loại class không đủ mẫu. | M1 §4.1. | KBC annotator nào đặt boundary, guideline boundary, xử lý đoạn có nhiều activity/distraction, và cách giải quyết chồng lấn. |
| **Annotation** | Bốn annotator được huấn luyện, nền tảng CS + physical therapy/intervention; mỗi người gán độc lập đúng một activity label/clip; majority vote cho nhãn cuối. | M1 §4.2; M0 §6.2. | KBC tie-break khi 2–2, annotation tool, blinding, confidence, Cohen/Fleiss κ/ICC, và nhãn quality/correctness/engagement. |
| **Preprocessing** | Từ clip RGB: Lucas–Kanade optical flow; OpenPose 2D multi-person; ROMP 2D/3D; nối demographic/clinical participant-level. Raw RGB không public. | M1 §4.3–4.5; M2; M3. | KBC identity tracking xuyên frame, chọn person nào là child, smoothing, statistical normalization, missing-pose repair, và alignment manifest giữa modality. |
| **Final sample** | Đơn vị khái niệm là **một activity clip** + một nhãn trong 11 class + các feature theo frame + clinical metadata ở cấp participant. Paper định dạng: flow `(L−1,H,W)`, 2D skeleton `(L,N,17,2)`, 3D `(L,N,24,3)`; release cục bộ thực tế có OpenPose BODY_25, ROMP 24-joint và một nhánh 71-joint. | M1 §4.4; M2; M3. | Phải version schema. Không có manifest authoritative trong workspace để xác nhận tập 1.315 sample và mapping 1:1 giữa mọi modality. |

### Kiểm tra artifact MMASD hiện có

- Paper công bố **1.315 clip / 244.679 frame**. Cây local không tái tạo đúng con số đó: optical-flow có 1.310 thư mục clip; các nhánh skeleton có 1.693 thư mục; chỉ 1.289 tên clip giao nhau giữa flow và skeleton.
- Trong local tree, 21 tên chỉ có ở flow và 404 tên chỉ có ở skeleton; phần lớn chênh lệch skeleton nằm ở `drumming`. Nhiều thư mục OpenPose rỗng. Số frame flow và skeleton cũng không bằng nhau; flow vốn có `L−1` nhưng một số chênh lệch lớn hơn 1.
- Vì vậy, **final sample dùng cho mô hình phải được dựng từ manifest đã kiểm tra**, không được lấy union thư mục. Thông tin còn thiếu chính xác là: release manifest/checksum chính thức, quy tắc ghép filename giữa modality, và danh sách 1.315 clip hợp lệ.

## 1.2 DRYAD

`Participant → Setup → Task → Recording → Segmentation → Annotation → Preprocessing → Final sample`

| Bước | Pipeline tái dựng | Bằng chứng | KBC / thông tin còn thiếu |
|---|---|---|---|
| **Participant** | Hai cohort chính ASD và TD; metadata age/gender/walking length ở cấp case. Nhánh `Severe level of ASD` là video-only riêng. | D1; D2. | KBC tiêu chuẩn/công cụ chẩn đoán ASD và việc 9 ca severe có trùng cohort chính hay không. |
| **Setup** | Kinect v2 cao khoảng 0,75 m, bật trước 30 phút; Samsung Note 9 rear camera cho color video; 20–22°C, 76–87 lux, tránh nắng trực tiếp. | D1. | KBC intrinsic/extrinsic calibration, hệ trục/toạ độ được công bố chính thức, hardware sync giữa Kinect và điện thoại, FPS/resolution chuẩn toàn bộ bộ dữ liệu. |
| **Task** | Trẻ đi dọc đường thẳng, tốc độ bình thường, hướng về Kinect; mỗi lượt 1,5–4 m và khoảng hai gait cycles. | D1. | KBC lời hướng dẫn chuẩn, điểm xuất phát/dừng, footwear, assistance và quy tắc xử lý trẻ không hoàn thành. |
| **Recording** | Color, Kinect skeleton, skeleton-movement video và joint-trajectory video được ghi 10 lần/participant. | D1. | KBC liệu cả 10 raw takes có được giữ; release chính chỉ cho thấy một `video/` đã chọn/case. |
| **Segmentation** | Chọn **một suitable gait cycle** từ các lượt; ngưỡng `(Max distance between feet + Min distance between feet)/2` được liệt kê là feature dùng để trích cycle. | D1. | KBC tiêu chí “suitable”, ai chọn, exact heel-strike/toe-off rule, frame boundary và quality exclusion. Không thể tái tạo selection chỉ từ README. |
| **Annotation** | Ground-truth public là group `A/ASD` hoặc `T/TD` lấy từ cohort/path. Không có annotation hành vi theo frame/trial. | D1; D2. | KBC provenance chẩn đoán và annotator; không có correctness, gait-quality hay symptom score. |
| **Preprocessing** | Kinect SDK tạo 25 joint 3D; mỗi selected cycle có workbook 3D, 2D, feature workbook và skeleton video. Tính tọa độ thống kê, 16 góc, khoảng cách, ROM, stride/gait timing/velocity; tạo 7 augmentation/subject. | D1; D2. | README local không liệt kê tên, công thức và tham số đầy đủ của bảy transformation; không có code chuẩn cho selection/augmentation/feature extraction. Smoothing/normalization/outlier rule KBC. Giá trị `100` trong 3 ASD original là sentinel có ý nghĩa **KBC**, không được tự impute. |
| **Final sample** | Có hai biểu diễn: (a) **sequence sample** = một selected gait cycle hoặc một augmentation, lưu 25×3 theo frame trong `.xlsx`; (b) **classifier-ready sample** = một hàng 1.259 feature + `class` trong `Final dataset.xlsx`. Cấu trúc: group → case → `video`/`augmentation/{1..7}`. | D1; D2; D3. | Không có official split. Augmentation phải giữ cùng participant/same duplicate component trong một partition. |

### Kiểm tra artifact DRYAD hiện có

- Toàn cây có 100 subject chính, mỗi subject có một `video/` và 7 augmentation; tổng **800 skeleton 3D workbook** đã được audit trong D3.
- Mỗi original/augmentation có 3 workbook (`3D`, `_2d`, `features`) và skeleton video; original còn có trajectory/color video. `Final dataset.xlsx` có 800 hàng nhưng 800 hàng không phải 800 người.
- D3 ghi nhận 44 nhóm numeric duplicate và 7 connected component chứa nhiều subject. Do đó lineage/duplicate component là metadata bắt buộc khi chia train/test.

## 1.3 DREAM

`Participant → Setup → Task → Recording → Segmentation → Annotation → Preprocessing → Final sample`

| Bước | Pipeline tái dựng | Bằng chứng | KBC / thông tin còn thiếu |
|---|---|---|---|
| **Participant** | Trẻ ASD được randomize vào RET/SHT; public record mang `participant.id`, `gender`, `ageInMonths`, ADOS pre/post khi có. | R1 §3–4; R3. | Không cần lặp count; KBC mapping participant tới dropout/reason trong JSON public. |
| **Setup** | Sensorized table; RET: NAO là interaction partner, therapist bên cạnh và supervisor phía sau; SHT: therapist là partner. Một số task dùng touchscreen sandtray, số khác dùng bàn/đồ vật. | R1 §3 và Fig. 1–2. | Exact setup thay đổi theo task nhưng public release không kèm scene layout, object coordinates hay calibration file cho từng session. |
| **Task** | Initial assessment → 8 intervention → final assessment. Mỗi intervention có 3–6 script parts. 12 scripts, ba abilities JA/IM/TT và difficulty tăng theo performance. Cấu trúc chung: stimulus → wait response → contingent feedback → retry. | R1 §3. | Full machine-readable script, exact utterance/action/object instance và task-state log không có trong public JSON. |
| **Recording** | Ba RGB + hai RGB-D ghi session; table tính child position/motion/gaze/face/verbal utterance/object location ở ~25 Hz. Raw/direct video, audio, depth không public. | R1 §3.1, §4; R2. | KBC exact audio channel/source, sensor clock drift và dropped-frame log. |
| **Segmentation** | Public record có `task.index`, `ability`, `difficultyLevel`, `start`, `end`; arrays trong một record được căn cùng độ dài. Filename/date nhóm các đoạn vào intervention/assessment. | R3; kiểm tra đại diện đủ mọi ability/difficulty trong R4. | KBC ai/thuật toán đặt boundary và liệu một JSON là script part, task episode hay unit kỹ thuật. Không được gọi một JSON là một discrete trial; public không có boundary từng attempt. |
| **Annotation** | Skeleton/head/gaze là perception tự động; condition/task lấy từ protocol. Clinical performance, correct/incorrect, engagement và face-expression không public; face-expression bị loại vì reliability thấp. | R1 §4; R3. | KBC human review của task boundary và signal; không có annotator agreement cho outcome. |
| **Preprocessing** | Kinect SDK cho 3D skeleton; POSIT head pose; relative eye gaze từ RGB-D, kết hợp head pose thành gaze 3D; sensor data đưa về joint Cartesian frame; mỗi joint có confidence, mỗi numeric stream cho phép `null`. | R1 Table 1; R2; R3. | Smoothing, interpolation, null handling, confidence calibration và exact multi-camera fusion/registration parameters KBC. |
| **Final sample** | Một JSON v1.2 chứa participant/ADOS, condition, task metadata, frame rate/time và các aligned arrays: eye-gaze vector, head-orientation vector, 10 upper-body joints 3D + confidence. Đây là **public segment record**, không phải raw session và không phải outcome-labeled trial. | R2; R3; R4. | Public release thiếu stimulus content, expected-response code, observed semantic response, assessment, feedback, retry/prompt marker, object/touch/robot state. |

### Kiểm tra artifact DREAM hiện có

- Audit metadata trên toàn bộ **3.121 JSON**: mọi record có task object với các khóa `index`, `ability`, `difficultyLevel`, `start`, `end`; condition gồm RET/SHT.
- Tuy nhiên 324 record có `ability=""`, `difficultyLevel=0`, `index=null`; 164 record có `end < start`, trong đó một record kiểm tra trực tiếp có `end=-136956183` nhưng 11.975 time steps hợp lệ trong mọi signal array. Vì vậy `end-start` không được tin tuyệt đối nếu chưa QC.
- Một số record hợp lệ có arrays rỗng hoàn toàn. Final loader phải dùng **độ dài mảng thực**, validate task metadata, giữ mask `null`/confidence, và flag/quarantine record bất thường; không âm thầm sửa `end`.

---

# 2. DREAM task reconstruction

## Quy tắc chung cho ba task

Trong intervention, partner (robot ở RET, therapist ở SHT) phát discriminative stimulus, chờ child response, rồi cho feedback phụ thuộc response: đúng/matched → positive feedback; dưới expectation → yêu cầu thử lại. Mỗi stimulus có tối đa ba attempts; nếu attempt cuối vẫn thất bại, therapist đưa behavioral prompt. **Pre/post-test không có feedback và không có prompt**. Nguồn: R1 §3, dòng protocol ngay trước mô tả từng task.

Public JSON chỉ nói record thuộc ability/difficulty nào; không chứa event marker cho bất kỳ mắt xích nào trong chuỗi trên.

## 2.1 Joint Attention (JA)

| Thành phần | Tái dựng có bằng chứng |
|---|---|
| **Stimulus** | Level 1: partner nhìn + chỉ vào picture object trên sandtray + nói “look”; level 2: nhìn + chỉ, không verbal cue; level 3: chỉ nhìn. (R1 §3.) |
| **Expected response** | Trẻ follow cue và nhìn đúng object được partner chỉ/nhìn. |
| **Observed behaviour** | Hệ thống gốc có eye gaze, head pose/orientation, skeleton và object tracking. Public chỉ giữ eye/head/skeleton; không giữ object location, pointing cue hay stimulus timestamp. |
| **Assessment** | Protocol coi đạt khi trẻ nhìn object được chỉ định; baseline M0 còn mô tả thang 0/1/2 nhưng score này không có trong public release. Không thể tính correctness từ public gaze vector một mình vì thiếu target/AOI và cue interval. |
| **Feedback** | Intervention: positive feedback nếu follow cue và nhìn đúng; dưới expectation thì indication to try again. Pre/post: không feedback. |
| **Retry** | Tối đa ba attempts/stimulus; fail lần cuối → therapist behavioral prompt. Không có attempt/prompt markers public. |
| **Recorded signals** | `eye_gaze.(rx,ry,rz)` là Cartesian heading vector; `head_gaze.(rx,ry,rz)` là head-orientation heading vector; `skeleton.head` cho vị trí đầu, các joint tay/vai có thể chứa chuyển động chỉ tay. Tất cả được align theo time step. |
| **Public fields** | `task={index, ability:"JA", difficultyLevel:1..3, start,end}`, `frame_rate`, `time/date`, gaze/head/skeleton arrays, participant/ADOS/condition. Không có `target_object`, `AOI`, `cue`, `look_onset`, `correct`, `latency`, `feedback`, `attempt`. |

**Liên hệ tín hiệu với JA:** eye gaze có quan hệ trực tiếp nhất với response “look at target”; head orientation là tín hiệu bổ trợ khi mắt thiếu/không tin cậy; skeleton có thể mô tả posture/hand movement nhưng paper không nói skeleton là criterion chấm JA. Paper nói sensor table dùng perception để hướng dẫn robot nói chung, nhưng **không công bố rule task-specific** kết hợp gaze/head/skeleton. Do đó chỉ được gọi đây là tín hiệu ứng viên, không phải nhãn JA.

## 2.2 Imitation (IM)

| Thành phần | Tái dựng có bằng chứng |
|---|---|
| **Stimulus** | Ba script/level: (1) partner làm cách dùng/chơi với object, ví dụ toy car; (2) common meaningful gesture, ví dụ vẫy tay + nói goodbye; (3) gesture không có nghĩa thông dụng. Partner làm trước rồi yêu cầu trẻ làm giống. |
| **Expected response** | Trẻ tái hiện chính xác hành vi mẫu tương ứng. |
| **Observed behaviour** | Hệ thống gốc có 3D skeleton/action recognition, object tracking, gaze, head pose, verbal utterance. Public giữ upper-body skeleton + gaze/head; không giữ object state, speech/audio, partner skeleton hay action-recognition label. |
| **Assessment** | Protocol đánh giá có imitate accurately hay không. Public không có action template, similarity score, partial/correct/no-response, hay therapist override. Với object imitation, chỉ skeleton trẻ là không đủ để xác nhận cách dùng object. |
| **Feedback** | Intervention: positive feedback nếu imitation chính xác; dưới expectation → try again. Pre/post: không feedback. |
| **Retry** | Quy tắc ba attempts và prompt chung; không có markers public. |
| **Recorded signals** | Skeleton có head/shoulder/elbow/wrist/hand trái-phải, nên liên hệ rõ nhất với upper-body gesture trajectory. Eye gaze/head orientation có thể phản ánh việc nhìn partner/object, nhưng tài liệu không nói chúng được dùng để score IM. |
| **Public fields** | `task.ability="IM"`, difficulty 1..3, index/start/end + aligned gaze/head/skeleton. Không có gesture ID, object ID, demonstration interval, partner pose, child-response interval, correctness hay feedback. |

**Giới hạn:** skeleton DREAM không có lower-body joints; vì vậy không được tổng quát “imitation” thành full-body imitation. Với common gesture có lời nói, public release cũng không cho phép đánh giá phần lời.

## 2.3 Turn-taking (TT)

| Thành phần | Tái dựng có bằng chứng |
|---|---|
| **Stimulus** | Ba loại game trên sandtray, mỗi loại hai level: (1) chia sẻ sở thích bằng chọn trong 5 pictogram, level khác nhau về độ phức tạp; (2) phân loại object: level 1 một object, level 2 tám object và chọn/move vào category; (3) hoàn thành pattern: chọn từ 2 rồi 4 picture. Partner và trẻ luân phiên thao tác. |
| **Expected response** | Trẻ chờ trong lượt partner, không chạm màn hình; đến lượt thì thực hiện move phù hợp với game. |
| **Observed behaviour** | Hệ thống gốc có object tracking và sandtray interaction context; public upper-body skeleton có thể cho thấy hand/wrist reach, còn gaze/head có thể cho thấy hướng chú ý. Public không có touch event, screen state, partner-turn interval hay selected pictogram. |
| **Assessment** | Good performance nếu trẻ chờ mà không chạm màn hình trong lượt partner. Correctness của lựa chọn/category/pattern không được phát hành. Không thể phân biệt “tay gần màn hình” và “đã touch” chỉ từ 10-joint skeleton. |
| **Feedback** | Positive feedback cho good performance; dưới expectation → try again. Pre/post: không feedback. |
| **Retry** | Quy tắc ba attempts/prompt chung; không có markers public. |
| **Recorded signals** | Hand/wrist/elbow skeleton là liên hệ chuyển động rõ nhất; eye gaze/head orientation chỉ là tín hiệu chú ý tiềm năng. Tài liệu không chỉ rõ feature/rule nào đã quyết định TT online. |
| **Public fields** | `task.ability="TT"`, difficulty 1..2, index/start/end + aligned gaze/head/skeleton. Không có `turn_owner`, `turn_start/end`, touch log, screen/object state, chosen response, correctness, wait violation, feedback/retry. |

## Kết luận riêng về gaze, head orientation và skeleton

| Signal | JA | IM | TT | Điều có thể khẳng định |
|---|---|---|---|---|
| **Eye gaze** | Gắn trực tiếp về mặt khái niệm với “nhìn đúng object”, nhưng thiếu target/AOI nên không tự tạo outcome. | Có thể mô tả child quan sát partner/object; không được tài liệu xác nhận là score criterion. | Có thể mô tả attention trong lượt; không có turn state để đối chiếu. | Chỉ DREAM public; vector hướng 3D liên tục, có `null`, không phải categorical gaze label. |
| **Head orientation** | Bổ trợ gaze/head-turn, đặc biệt khi eye gaze thiếu; không thay thế target label. | Có thể mô tả orientation khi quan sát/làm gesture; không có scoring rule. | Có thể mô tả hướng chú ý tới partner/sandtray; không có scoring rule. | `head_gaze` là Cartesian heading vector, không phải trực tiếp yaw/pitch/roll hay outcome. |
| **Skeleton** | Cho head position và cử động tay/vai; không đủ biết object trẻ nhìn. | Tín hiệu phù hợp nhất cho upper-body gesture imitation; thiếu partner pose/object/action label. | Có thể phát hiện reach/wait candidate; không đủ xác nhận touch/turn ownership. | 10 joint upper body + confidence; không có lower body, object, partner hay robot pose public. |

---

# 3. Cross-dataset mapping

Phân loại dùng bốn nhãn: **Trực tiếp**, **Một phần**, **Chỉ một số bộ**, **Không nên chuẩn hóa chung**. “Trực tiếp” chỉ có nghĩa cùng loại entity/khóa, không có nghĩa phân bố hay ý nghĩa lâm sàng giống nhau.

| Tầng | MMASD | DRYAD | DREAM | Phân loại | Mapping an toàn |
|---|---|---|---|---|---|
| **Participant** | ID trong filename + clinical workbook | `group/case_id` | `participant.id` | **Tương đương trực tiếp** | `dataset_id + participant_id`; tuyệt đối namespace ID theo dataset. Clinical/group để ở attributes. |
| **Session** | Buổi trị liệu nguồn; public chủ yếu còn clip, day code trong filename chưa có grammar chính thức | 10 acquisition takes, không phải therapy session; release giữ selected cycle | Assessment/intervention; nhiều JSON có thể thuộc cùng intervention | **Không nên chuẩn hóa chung** | Dùng `recording_event` + `event_type={therapy_session, gait_take, assessment, intervention}`; không ép một JSON/clip/cycle = session. |
| **Task** | Một trong 11 activity | Một walking instruction | Ability JA/IM/TT + difficulty/script | **Tương đương một phần** | `task_family`, `task_name`, `difficulty`, `protocol_task_id`; giữ taxonomy gốc. |
| **Segment** | Clip activity cắt thủ công | Selected gait cycle hoặc augmentation | JSON public segment có task start/end | **Tương đương một phần** | `segment_id`, parent recording, boundary provenance, segment_type. Không coi segment = trial. |
| **Frame–Time** | Frame index, source 25–30 FPS; flow thường L−1 | Workbook có time/frame, 21–60 frame observed trong skeleton audit | Arrays aligned, `frame_rate`, start/end; có metadata lỗi | **Tương đương một phần** | Quy về `timestamp_seconds` chỉ khi FPS/time hợp lệ; luôn giữ `frame_index`, `valid_length`, `clock_source`, QC flag. |
| **Skeleton** | OpenPose 25-joint 2D; ROMP 24/71-joint 2D/3D; multi-person | Kinect 25-joint 3D, một child | 10-joint upper-body 3D + confidence | **Tương đương một phần** | Một canonical core upper-body graph + `source_topology`, `coord_system`, `person_role`, confidence/mask. Không hallucinate joint thiếu. |
| **Movement** | Optical flow + activity label + pose | Gait trajectory, angles, distances, ROM, spatiotemporal gait features | Motion suy từ upper-body position; không movement/action label public | **Tương đương một phần** | Chuẩn hóa raw kinematics (position/velocity) khi có; giữ `activity`, `gait_feature`, `therapy_skill` là ba namespace khác nhau. |
| **Gaze** | Không có | Không có | Eye-gaze/head-orientation vector 3D | **Chỉ có ở một số dataset** | Trường nullable với `gaze_vector`, `head_vector`, `target/AOI` tách riêng. Không biến head joint thành gaze. |
| **Stimulus** | Trainer/model/robot làm mẫu nhưng không có event public | Verbal instruction “walk normally” ở protocol, không log | Scripted discriminative stimulus, không có event/content public | **Không nên chuẩn hóa chung** | Chỉ schema mới có thể dùng generic `stimulus_event`; dữ liệu cũ không đủ để backfill timestamp/content. |
| **Response** | Child activity movement, không có semantic response/correctness | Gait recording, không phải therapy response | Child signals có nhưng response event/label bị thiếu | **Không nên chuẩn hóa chung** | Phân biệt `observed_signal` với `coded_response`. Không đổi motion thành response outcome. |
| **Outcome** | Activity class không phải kết quả | ASD/TD class không phải task outcome | Performance/correctness withheld | **Không nên chuẩn hóa chung** | Không map activity/diagnosis/ability vào một cột `label`. Nếu xây mới: outcome riêng theo trial/session/clinical level. |
| **Metadata** | Demographic + ADOS/motor/severity; filename activity/day | Demographic + group/walking length + lineage | Demographic + ADOS + RET/SHT + date/task/fps | **Tương đương một phần** | Core: provenance, participant, age unit, gender vocabulary, recording/task IDs; extension theo dataset cho clinical/condition/augmentation. |

## Canonical hierarchy đề xuất khi cần hợp nhất

```text
Dataset
└── Participant
    └── RecordingEvent (typed; không mặc định là “session”)
        └── TaskOccurrence (taxonomy gốc + protocol/version)
            └── Segment (clip / gait_cycle / public_task_segment)
                ├── TimeBase + QC mask
                ├── Signal streams (skeleton / flow / gaze ...)
                ├── Stimulus events [nullable, provenance-required]
                ├── Response events [nullable, provenance-required]
                └── Outcome labels [nullable, level-required]
```

Không tạo một cột `label` chung. Ít nhất phải tách `activity_label`, `diagnosis_group`, `therapy_ability`, `response_outcome` và `treatment_condition`.

---

# Nếu tự xây một dataset trị liệu ASD thì học gì?

## Nên học từ MMASD

- Thu trong bối cảnh trị liệu tự nhiên, tương tác child–trainer–model, hoạt động vận động đa dạng.
- Cắt clip theo boundary activity, nhiều annotator độc lập, và privacy-first release bằng flow/pose thay vì raw face video.
- Giữ metadata lâm sàng ở cấp participant nhưng nối bằng ID/manifest rõ ràng.
- Không sao chép điểm yếu: phải công bố boundary guideline, tie-break, agreement, child identity tracking, confidence/missing mask và manifest modality hoàn chỉnh.

## Nên học từ DRYAD

- Protocol vận động kiểm soát, nhiều take trước khi chọn gait cycle, skeleton toàn thân Kinect 25 joint và feature gait có ý nghĩa cơ sinh học.
- Nếu mục tiêu có so sánh motor phenotype, dùng control phù hợp và ghi environmental/calibration metadata.
- Mọi augmentation phải có parent ID, transformation + parameter và split theo participant/duplicate component.
- Không sao chép điểm yếu: không dùng “suitable cycle” nếu không có rule; không để sentinel/duplicate không giải thích.

## Nên học từ DREAM

- Thiết kế task bằng script và difficulty; log rõ `stimulus → response → assessment → feedback → retry/prompt`.
- Dùng RET/SHT tương đồng về protocol, supervised autonomy, common coordinate system, time-aligned multimodal streams, confidence và null mask.
- Tách dữ liệu nhận dạng khỏi public secondary features, nhưng vẫn phát hành event/outcome an toàn để dữ liệu có giá trị trị liệu.

## Những thành phần quan trọng cả ba bộ vẫn thiếu hoặc không phát hành đầy đủ

1. **Event log trial-level hoàn chỉnh:** timestamp và nội dung stimulus, response window, observed response, assessment, feedback, retry/attempt, prompt và therapist override.
2. **Outcome công khai có provenance:** correct/partial/no-response, latency, movement quality, engagement, assessor ID, confidence và inter-rater agreement. Không bộ nào public trọn vẹn chuỗi này.
3. **Gaze có ngữ nghĩa:** target/AOI/object/partner ID, object coordinates, fixation onset/duration và calibration quality. DREAM chỉ có vector; hai bộ kia không có gaze.
4. **Interaction/task state đồng bộ:** robot command/state, therapist action, partner pose, sandtray/touch/object events và speech transcript/VAD. Đây là phần cần để giải thích vì sao trẻ phản ứng.
5. **Raw-to-derived provenance tái lập:** sensor clock/calibration, detector/model version, person tracking, transformation parameters, missing/outlier repair, checksum và manifest xuyên modality.
6. **Quality/reliability chuẩn:** per-frame confidence cho mọi modality, QC flag, rule loại sample, agreement và audit trail. DREAM mới chỉ làm tốt một phần với joint confidence/null.
7. **Đánh giá dọc và khái quát hóa:** outcome session-level, change-from-baseline, maintenance/transfer sang người và bối cảnh khác, treatment fidelity và nguyên nhân dropout.
8. **Official participant-independent benchmark split:** train/validation/test hoặc cross-validation khóa theo participant và lineage; cả ba đều chưa có chuẩn chung chống leakage.

## Kết luận một câu

Dataset mới nên lấy **bối cảnh vận động tự nhiên + privacy-preserving feature/annotation của MMASD**, **độ chặt của acquisition gait và lineage augmentation của DRYAD**, và **protocol ABA theo event, difficulty, multimodal alignment/confidence của DREAM**; phần cần tự bổ sung quan trọng nhất là một event/outcome layer machine-readable theo từng trial, vì đây chính là mắt xích cả ba public dataset đều chưa cung cấp đầy đủ.
