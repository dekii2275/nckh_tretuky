# Đối chiếu MMASD – DRYAD – DREAM

Ngày kiểm tra: 30/09/2026  
Phạm vi: tài liệu gốc/paper, schema/README và dữ liệu đang có trong workspace.

## Cách đọc báo cáo

- **Có / Không** chỉ được dùng khi nguồn cho phép kết luận rõ.
- **Không báo cáo (KBC)** nghĩa là paper, README, schema và file public không đủ bằng chứng; không đồng nghĩa chắc chắn là thí nghiệm không có.
- Với DREAM, cần phân biệt **hệ thống thu thập gốc** (có video/audio và nhiều perception) với **bản public** (chỉ có feature thứ cấp và metadata).
- Với MMASD, “session” là buổi can thiệp khoảng 45 phút; “sample” public là clip hoạt động ngắn được cắt từ session, không phải một session hay một discrete trial.
- Với DRYAD, đây là thí nghiệm dáng đi, không phải trị liệu. “800 samples” gồm 100 mẫu gốc và 700 bản augmentation, không phải 800 trẻ.

## Kết luận nhanh

| Bộ dữ liệu | Phù hợp nhất cho | Không phù hợp nếu cần |
|---|---|---|
| **MMASD** | Nhận diện 11 hoạt động trị liệu, phân tích vận động, chất lượng động tác và đồng bộ vận động từ optical flow/skeleton | Nhãn stimulus–response–feedback, gaze, emotion, audio, raw RGB công khai |
| **DRYAD** | Phân loại ASD/TD và phân tích dáng đi/chuyển động toàn thân bằng Kinect v2 | Robot therapy, joint attention, imitation, turn-taking, engagement hoặc giao tiếp xã hội |
| **DREAM** | Phân tích hành vi trong trị liệu robot so với trị liệu người; skeleton–head gaze–eye gaze theo thời gian; ba kỹ năng xã hội | Raw video/audio công khai, nhãn đúng/sai từng trial, emotion/engagement public |

---

## 1. Mục tiêu dataset

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Bài toán chính** | Benchmark đa phương thức bảo vệ riêng tư cho phân tích vận động và hành vi xã hội trong play/physical therapy; hỗ trợ activity recognition, action-quality assessment, interpersonal synchrony và theo dõi trạng thái/phát triển. Không phải bộ điều khiển robot. | Phân loại **ASD so với phát triển điển hình (TD)** và phân tích khác biệt dáng đi/chuyển động toàn thân từ Kinect v2. Không phải dataset trị liệu. | Hỗ trợ nghiên cứu data-driven về ASD và so sánh **Robot-Enhanced Therapy (RET)** với **Standard Human Treatment (SHT)** theo ABA; feature thời gian từng được dùng online để hỗ trợ robot, nhưng bản public chủ yếu phục vụ phân tích. |
| **Mục tiêu trị liệu** | Thúc đẩy eye contact, body gesture/balance, phối hợp, fine motor và interpersonal synchrony qua robot movement, rhythm và yoga. | Không huấn luyện kỹ năng trị liệu; trẻ chỉ đi bộ bình thường về phía camera. | Huấn luyện/đánh giá ba kỹ năng: **imitation, joint attention, turn-taking**. |
| **Joint Attention** | Không có task/label joint-attention chính thức trong 11 lớp. Bối cảnh tam giác có thể tạo cơ hội chú ý chung nhưng không được operationalize thành trial/nhãn. | Không. | Có. Ba mức: (1) nhìn + chỉ + lời “look”; (2) nhìn + chỉ, không lời; (3) chỉ nhìn. Đúng khi trẻ theo cue và nhìn đúng đối tượng trên sandtray. |
| **Imitation** | Có ở mức hoạt động: trẻ làm theo robot/trainer/model, chủ yếu bắt chước vận động cơ thể, nhịp điệu và tư thế; không có nhãn riêng cho biểu cảm/âm thanh. | Không. | Có ba script: bắt chước dùng đồ vật, cử chỉ có nghĩa và cử chỉ không có nghĩa; protocol/deliverable còn mô tả vận động + âm thanh và cử chỉ cảm xúc, nhưng bản public không chứa score đúng/sai. |
| **Turn-taking** | Không có task/annotation turn-taking rõ ràng. | Không. | Có. Ba nhóm game trên sandtray: sở thích, phân loại và hoàn thành pattern; đánh giá trẻ biết chờ, không chạm màn hình trong lượt của partner. |
| **Emotion Recognition** | Không. Paper nói chất lượng video không đủ cho feature tinh tế như biểu cảm mặt. | Không. | Hệ thống gốc từng tính facial expression, nhưng bị loại khỏi bản public do độ tin cậy thấp. Không có nhãn emotion public. |
| **Engagement** | Không có nhãn engagement public. | Không. | Protocol có thang engagement 0–5 (intense noncompliance → intense engagement), nhưng biến performance/clinical evaluation này **không được phát hành** trong dataset public. |
| **Social Communication** | Được nhắm đến gián tiếp qua eye contact, phối hợp và synchrony; không có label giao tiếp xã hội riêng. | Không. | Có ở cấp mục tiêu trị liệu qua joint attention, imitation và turn-taking; tuy nhiên public release chỉ cho task/condition và feature chuyển động–gaze, không có outcome từng trial. |

## 2. Đối tượng tham gia

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Số lượng** | Paper/repo: **32 trẻ ASD** (27 nam, 5 nữ). Audit `ADOS_rating.xlsx`: 36 dòng, tương ứng **35 ID gốc duy nhất** sau khi gộp `40023_R`; do không có toàn bộ clip cục bộ nên không xác định 32 ID nào thuộc release. Khi báo cáo benchmark nên dùng N=32 và ghi chênh lệch workbook. | Cohort chính: **100 trẻ** = 50 ASD + 50 TD. Có thêm thư mục video-only của **9 ca ASD nặng** (10 video); tài liệu không làm rõ 9 ca này có trùng cohort chính hay không, vì vậy không nên khẳng định N=109. | 76 tuyển; 70 đạt tiêu chí và random hóa; 9 dừng sau đánh giá đầu; **61 trẻ trong public study/release** (RET 30, SHT 31; 9 nữ). Workspace có đúng 61 thư mục người dùng. |
| **Độ tuổi** | Công bố: 5–12 tuổi, không báo mean/SD. Workbook cục bộ 35 ID: 59–151 tháng (4,92–12,58 tuổi), **7,90 ± 2,33 tuổi** (SD mẫu); đây không chắc là đúng subset 32 trẻ public. | Metadata 100 trẻ: TD 2–14 tuổi, **6,66 ± 3,08**; ASD 5–15 tuổi, **8,62 ± 2,19** (SD mẫu, tính lại từ 50 dòng mỗi nhóm). README có bảng mean theo giới nhưng không báo SD. | Công bố 3–6 tuổi. Audit một record/participant trong workspace: 61 trẻ, 36–72 tháng, **52,89 ± 11,68 tháng** = **4,41 ± 0,97 tuổi**. |
| **Đặc điểm ASD** | Screening bằng SCQ; eligibility bằng ADOS-2 và clinical judgment. Có ADOS comparison score, motor-function level và severity tổng hợp. | Chỉ ghi “children with ASD” và một nhóm “severe level”; không nêu tiêu chí chẩn đoán hay cách gán mức nặng. | Chẩn đoán ASD; public cohort có ADOS ban đầu 7–20. |
| **Đánh giá lâm sàng** | SCQ, ADOS-2 (module, Social Affect, RRB, total, classification, comparison score), Vineland Adaptive Behavior Scales; motor functioning 1–3. | Không thấy ADOS/ADI-R/CARS/DSM hoặc thang lâm sàng trong README/file public. | ADOS-G/ADOS (module, communication, interaction, play, stereotype, total) và Social Communication Questionnaire trong schema; initial/final assessment dùng ADOS, nhưng public chủ yếu có initial score. |
| **Nhóm đối chứng** | Không; paper nêu tương lai nên bổ sung trẻ TD ghép tuổi/giới. “Therapist/model làm cùng hoạt động” là template vận động, không phải control cohort. | Có **50 TD**, cân bằng với 50 ASD ở cohort chính. | Không có TD. Có nhóm đối chứng điều trị **SHT (n=31)** so với RET (n=30), tất cả đều ASD. |
| **Số session/người** | Lịch nghiên cứu 10 tuần, expert training 4 buổi/tuần, khoảng 45 phút/buổi; tương đương khoảng 40 buổi theo lịch, nhưng số thực tế/người không báo. Có thêm khuyến nghị 2 buổi practice/tuần với cha mẹ. | Camera ghi 10 lượt đi/người, sau đó chọn một gait cycle phù hợp; release chính có 1 mẫu gốc/người + 7 augmentation. Không có therapy session. | Protocol: initial assessment + 8 interventions + final assessment. 41/61 hoàn tất cả 8 + final; người còn lại trung bình 4,5 interventions. Public có 3.121 JSON/segment trên 61 người, không phân đều. |
| **Thời lượng** | Session khoảng 45 phút. Public clip: trung bình **7,0 ± 3,4 s** (paper cũng nêu 7,15 s); clip <3 s bị loại. | Mỗi lượt trẻ đi 1,5–4 m, khoảng hai gait cycles; một cycle được cắt ra. Không báo duration chuẩn; video mẫu audit dài 1,483 s nên không được khái quát cho cả bộ. | Một intervention 3–87 phút, median 32 phút; trial registry mô tả mục tiêu khoảng 45 phút. |
| **Điều kiện tham gia** | Tuyển qua trường học/dịch vụ/nhóm self-advocacy; SCQ screening, ADOS-2 + clinical judgment; 5–12 tuổi. Không nêu exclusion cụ thể. | Không nêu inclusion/exclusion ngoài nhãn ASD/TD và khả năng hoàn thành task đi bộ. | ASD, 3–6 tuổi, đủ điều kiện clinical trial; 70/76 đạt tiêu chí. Paper/record công khai không liệt kê chi tiết toàn bộ exclusion; một số trẻ dừng vì performance kỹ năng đã cao. |

## 3. Bối cảnh thí nghiệm/trị liệu

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Không gian** | Môi trường nhà ở của trẻ, bối cảnh tự nhiên với nền, ánh sáng và góc nhìn thay đổi. | Môi trường kiểm soát tại University of Babylon; 20–22°C, 76–87 lux, camera tránh nắng trực tiếp. | Ba địa điểm tại Romania; phòng trị liệu bố trí sensorized table/sandtray, camera và Kinect. |
| **Người tương tác** | Child + expert trainer + adult model/confederate; expert là PT hoặc học viên PT/kinesiology được huấn luyện nhi khoa; đôi khi có robot và cha mẹ practice ngoài expert sessions. | Researcher/camera operator; không có therapist hay robot trong task chính. | RET: trẻ tương tác với NAO, một therapist bên cạnh và một supervisor phía sau; SHT: trẻ tương tác trực tiếp với therapist, không có supervisor thứ hai. |
| **Vai trò robot** | Chỉ trong theme Robot: làm mẫu/hướng dẫn vận động để trẻ theo/bắt chước. Không phải nguồn ghi dữ liệu hay robot tự học. | Không có robot. | NAO là interaction partner/mediator: phát discriminative stimulus, thực hiện hành động theo script và đưa feedback/reinforcement dưới supervised autonomy. |
| **Vai trò therapist** | Trainer/model trực tiếp dẫn hoạt động và làm mẫu; paper không mô tả điều khiển robot theo từng lệnh. | Không có vai trò trị liệu. | RET: therapist giám sát/điều chỉnh diễn giải và phản ứng robot khi cần; SHT: therapist trực tiếp chạy script. Pre/post-test không prompt hay feedback. |

## 4. Kịch bản tương tác

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Cấu trúc session** | 10-week intervention; mỗi buổi có welcoming/warm-up, các hoạt động theo theme, và debriefing. Hai pha đầu/cuối tạo mốc thời gian hỗ trợ cắt dữ liệu. | Khởi động Kinect 30 phút; trẻ đi thẳng tốc độ thường về phía Kinect, quay 10 lượt, chọn một gait cycle rồi trích đặc trưng/augmentation. | Initial assessment → 8 interventions → final assessment. Mỗi intervention chia 3–6 phần theo script; 12 script, nhiều exercise và mức khó. |
| **Cấu trúc trial** | Không định nghĩa discrete trial. Đơn vị public là đoạn từ start đến end của một hoạt động được annotator cắt thủ công. | Một acquisition take là lượt đi 1,5–4 m; dữ liệu cuối dùng một gait cycle được chọn. | Stimulus → chờ response → contingent feedback. Mỗi stimulus tối đa 3 attempts; sau lần cuối thất bại therapist prompt. Bản public không phát hành boundary/response của từng attempt một cách đầy đủ. |
| **Stimulus** | Trainer/model hoặc robot làm mẫu; dụng cụ gồm robot, trống, maracas, bài hát và hướng dẫn yoga. | Yêu cầu bằng lời của researcher: đi bình thường theo đường thẳng hướng camera. | Robot hoặc therapist đưa chỉ dẫn/hành động; object/toy/pictogram/sandtray tùy imitation, joint attention hay turn-taking. |
| **Expected response** | Thực hiện hoạt động mục tiêu: 11 lớp vận động/nhịp/yoga. | Đi bình thường để thu gait cycle. | Imitate hành vi; nhìn đúng object theo cue; hoặc thực hiện/chờ đúng lượt trên sandtray. |
| **Correct response** | Không có score correctness public; chỉ có activity class. | Không có đúng/sai lâm sàng; take phù hợp là gait cycle đủ dùng được researcher chọn. | Khớp hành vi mong đợi: imitation chính xác; follow cue và nhìn object; turn-taking đúng, đặc biệt chờ mà không chạm khi partner đang đi. |
| **Incorrect/no response** | Không định nghĩa nhãn; clip nhiễu/ngắn/occluded có thể bị loại ở preprocessing. | Không định nghĩa label incorrect; take không phù hợp không được chọn. | Performance dưới expectation hoặc không phản ứng; protocol grid thường dùng 0 = không phản ứng/làm việc khác, 1 = cố gắng/partial, 2 = đạt, nhưng các score này không có trong public release. |
| **Feedback** | KBC theo từng trial. | Không áp dụng. | Positive feedback khi đúng; yêu cầu thử lại khi chưa đạt. Pre/post-test: không feedback. |
| **Retry** | KBC. | Quay 10 lượt rồi chọn một, nhưng không phải retry trị liệu theo correctness. | Có, tối đa 3 attempts/stimulus; sau thất bại cuối therapist cung cấp behavioral prompt. |
| **Số trial/session** | KBC; 1.315 clip là toàn bộ release, không phải trial count/session. | 10 recording takes/người; một gait cycle được giữ làm mẫu gốc. | KBC số discrete trials; intervention gồm 3–6 script parts. |
| **Randomization** | Không báo random hóa hoạt động/stimulus. | Không báo random hóa. | Random hóa trẻ vào RET/SHT. Tài liệu protocol cho biết thứ tự ba target abilities được random hóa giữa trẻ; stimulus cụ thể không được mô tả là random, mức khó tăng theo performance. |

## 5. Thiết bị thu thập

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **RGB camera** | Một RGB camera hướng vào trẻ. Clip nguồn 320×240 đến 1920×1080, 25–30 FPS; model/trainer có thể cùng khung hình. | Kinect v2 + camera sau Samsung Note 9 cho màu; Kinect cao 0,75 m. Tài liệu không cho cấu hình chung về resolution/FPS. Audit một `video.avi`: 736×1680, 60 FPS, MPEG-4—chỉ là một file mẫu. | Ba RGB camera độ phân giải cao; deliverable ghi 1280×960, bố trí trong sensorized table để quan sát mặt/gaze/chuyển động. |
| **RGB-D** | Không; 3D pose suy ra từ RGB đơn bằng ROMP. | Có Microsoft Kinect v2; depth/skeleton dùng để sinh tọa độ 3D nhưng raw depth map không thấy trong release. | Có hai Kinect RGB-D. |
| **Microphone** | KBC; không có audio public. | Không mô tả; không có audio file riêng và video mẫu audit chỉ có stream video. | Hệ thống gốc có verbal-utterance perception và NAO có microphone, nhưng số/nguồn mic thu liệu không nêu rõ trong dataset paper; public release không có audio. |
| **Eye tracking** | Không. | Không. | Eye gaze được **ước lượng từ camera/Kinect**, không phải eye tracker đeo người; public là vector 3D. |
| **Skeleton** | 2D bằng OpenPose; 3D bằng ROMP từ monocular RGB. | Kinect v2 skeleton, 25 khớp 3D. | Hai Kinect + sensor processing tạo upper-body 3D skeleton trong hệ tọa độ chung. |
| **Physiological sensors** | Không báo ECG/EDA/HR/EEG. | Không. | Không; nghiên cứu chủ động tránh wearable/on-body sensors. |
| **Robot logs** | Không public. | Không áp dụng. | Public chỉ có `condition`, task và time; không có chuỗi command/action/state chi tiết của NAO. Source code hệ thống có công bố riêng. |
| **Therapist logs** | Có training diary do cha mẹ phối hợp expert trainer, nhưng diary không phải modality public; không mô tả realtime annotation. | Không báo. | Performance/clinical variables và therapist assessment không phát hành; không có note realtime trong JSON public. |

## 6. Dữ liệu raw

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Video** | Có khi thu thập (>108 h) nhưng **không public** vì riêng tư. Public chỉ có feature bảo vệ riêng tư. | Có RGB `.avi`, skeleton video và trajectory video trong release, gồm cả mặt trẻ; cần xử lý như dữ liệu nhạy cảm. | Có trong nghiên cứu gốc nhưng **không public** theo consent/ethics. |
| **Audio** | Không public/không xác nhận đã thu. | Không có stream audio trong video mẫu; không thấy audio modality riêng. | Có direct audio measurement ở nghiên cứu gốc nhưng không public. |
| **Depth** | Không. | Kinect dùng depth để tạo skeleton, nhưng raw depth maps không thấy trong release. | Kinect depth được thu ở hệ thống gốc nhưng raw depth không public. |
| **Sensor stream** | Public: optical-flow frames, 2D/3D skeleton theo frame; không phải raw sensor. | Tọa độ 3D joint theo frame/time trong `.xlsx`, cùng engineered features; video là raw/derived tùy loại. | Public: aligned secondary streams ở khoảng 25 Hz: 10 upper-body joints + confidence, head-gaze/pose và eye-gaze vectors. |

## 7. Feature được trích xuất

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Face** | Không có face box/landmark public. | Không. | Hệ thống gốc có face detection/alignment; không có face box/landmark trong JSON public. |
| **Head pose** | Không. | Tọa độ joint Head/Neck có, nhưng không có yaw/pitch/roll. | Có `head_gaze` vector `(rx, ry, rz)`/3D head orientation; không lưu trực tiếp tên yaw/pitch/roll. |
| **Eye gaze** | Không. | Không. | Vector hướng gaze 3D `(rx, ry, rz)`, không phải categorical target gaze. |
| **Skeleton** | Bản release/repo hiện tại: OpenPose BODY_25 = **25 khớp 2D** và ROMP output **71 khớp 3D** (sample audit: `(3 people,71,3)`). Paper v3 lại mô tả COCO 17 và ROMP 24; phải ghi version khi dùng. | **25 khớp 3D** Kinect; còn có file 2D và 16 góc khớp/đặc trưng gait. | **10 khớp upper body 3D**: head, shoulder center/L/R, elbow L/R, wrist L/R, hand L/R; mỗi joint có confidence. |
| **Gesture/action** | Một nhãn clip trong 11 action classes: arm swing, body swing, chest expansion, squat, drumming, maracas forward shaking, maracas shaking, sing-and-clap, frog/tree/twist pose. | Không có action taxonomy; chỉ nhãn ASD/TD, gait/body features. | Task ability JA/IM/TT và difficulty; không có action/correctness label public. |
| **Speech** | Không ASR/transcript/VAD. | Không. | Hệ thống gốc xử lý verbal utterances, nhưng public không có audio/ASR/transcript/VAD. |
| **Emotion** | Không. | Không. | Facial-expression estimate bị loại khỏi public vì reliability thấp. |
| **Robot state** | Không. | Không áp dụng. | Không có pose/command/task-state đầy đủ; chỉ condition và task metadata. |

## 8. Biểu diễn thời gian

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Timestamp** | Dữ liệu clip theo frame; không thấy timestamp tuyệt đối cho từng modality trong public release. Start/end dùng nội bộ để cắt clip. | Skeleton workbook có cột time/frame; không chứng minh video và mọi file có timestamp đồng nhất. | Có ngày/giờ session, `frame_rate`; array đồng chỉ số frame. Schema v1.2 có `task.start/end` theo frame. Không có timestamp tuyệt đối cho từng sample của từng modality. |
| **FPS / sampling rate** | 25–30 FPS tùy video. | Không báo dataset-wide; raw sequence 21–60 frame/gait cycle trong audit. Không nên suy từ một video 60 FPS thành toàn bộ bộ. | Thu realtime 25 Hz; JSON có `frame_rate` (ví dụ 25,1). |
| **Synchronization** | Các feature đều dẫn xuất từ cùng RGB frame nên đồng bộ theo frame về nguyên tắc; không mô tả clock/hardware sync. | Video/skeleton cùng acquisition nhưng phương pháp sync không báo. | Sensorized table đưa dữ liệu vào hệ tọa độ Cartesian chung và temporal stream khoảng 25 Hz; chi tiết hardware clock không báo. |
| **Event markers** | Không có marker stimulus/response/feedback; chỉ boundary clip hoạt động. | Không. | `task.start/end` đánh dấu segment task ở schema 1.2; không có marker public riêng cho stimulus, response, feedback và retry. |

## 9. Annotation – đơn vị gán nhãn

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Frame-level** | Skeleton/flow là frame-level feature nhưng activity label áp dụng cho cả clip, không phải label thủ công từng frame. | Tọa độ skeleton theo frame; class A/T áp dụng toàn sample/người. | Skeleton/head/eye stream theo frame; không có behavior/correctness label từng frame. |
| **Segment-level** | Có: clip activity được cắt thủ công theo start/end và gán một lớp. | Không có segment behavior label. | Có task segment `start/end` trong schema 1.2; ability/difficulty áp dụng segment. |
| **Trial-level** | Không có trial/correctness label; clip có thể dùng như analysis unit. | Nhãn A/T ở mức selected gait-cycle sample; 7 augmentation kế thừa nhãn. | Protocol có trial/attempt nhưng public release không có result từng trial. |
| **Session-level** | Clinical/demographic score theo participant, không theo session; clip label theo activity. | Class theo participant/sample, không có session score. | Có condition, task/date, participant/ADOS trong mỗi JSON; không có outcome score toàn session public. |

## 10. Annotation – loại nhãn

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Behaviour** | 11 activity labels; không có behavior state chi tiết. | ASD (`A`) / TD (`T`), không phải hành vi. | `JA`, `IM`, `TT` + difficulty/condition; không có hành vi con public. |
| **Gaze** | Không. | Không. | Có vector liên tục, không có label nhìn robot/người/vật thể/khác. |
| **Action** | 11 hoạt động nêu trên. | Một task walking; không có action labels đa lớp. | Task ability, không có gesture/action label từng lần. |
| **Response** | Không có correct/incorrect/no-response. | Không áp dụng. | Protocol định nghĩa nhưng public dataset không phát hành response label. |
| **Engagement** | Không. | Không. | Protocol 6 mức 0–5, nhưng không có trong public JSON. |
| **Emotion** | Không. | Không. | Không public; face expression bị loại vì reliability thấp. |
| **Interaction state** | Không có stimulus/response/feedback/retry state. | Không. | Task boundary/condition có; state machine chi tiết và feedback/retry marker không public. |

## 11. Quy trình annotation

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Người gán nhãn** | Bốn annotator được huấn luyện, có nền tảng liên ngành computer science và physical therapy/intervention. | Không có manual behavior annotation được mô tả; nhãn diagnosis/group lấy từ cohort. | Public feature chủ yếu được sinh tự động bởi sensor system; task/condition đến từ protocol/script. Không công bố manual annotator cho public JSON. |
| **Số annotator** | Mỗi clip được cả **4 người** annotate độc lập. | KBC/không áp dụng. | KBC/không áp dụng cho public release. |
| **Guideline/codebook** | Paper mô tả 11 lớp bằng câu chữ, nhưng không phát hành annotation manual chi tiết. | Không có codebook behavior; README giải thích class A/T và engineered features. | JSON Schema là data dictionary tốt; protocol/deliverable mô tả task và behavioral grids, nhưng các performance labels không nằm trong public data. |
| **Công cụ** | KBC công cụ annotation. | Không áp dụng/KBC. | KBC công cụ annotate; có DREAM Data Visualizer để xem lại, không phải công cụ gán nhãn. |
| **Blind annotation** | Không báo annotator có blind participant/condition hay không. | Không báo. | Không báo; task/condition là metadata hệ thống chứ không phải blinded rating. |
| **Trường hợp ambiguous** | Final label bằng **majority vote**. Clip <3 s, chất lượng kém, ánh sáng/occlusion nặng bị loại. | Không nêu quy tắc ambiguous; researcher chọn một gait cycle “suitable”. | Public data không có consensus manual; system supervisor có thể sửa diễn giải online của robot khi cần. |

## 12. Chất lượng annotation

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Inter-rater agreement** | Không báo hệ số agreement dù có 4 annotator. | Không báo/không áp dụng. | Không báo cho public dataset. |
| **Cohen's κ** | Không. | Không. | Không. |
| **ICC/Fleiss κ** | Không. | Không. | Không. |
| **Consensus** | Majority vote giữa 4 annotator. Paper không nói cách xử lý hòa 2–2. | Không báo. | Không báo; face-expression feature bị loại do reliability thấp thay vì phát hành. |

## 13. Preprocessing

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Detection** | OpenPose multi-person cho pose 2D; ROMP cho multi-person 3D; Lucas–Kanade cho optical flow. Không phát hành face/object detector. | Kinect v2 hardware/SDK tracking cho skeleton; Samsung Note 9 cho RGB. Không có face/object detection. | Sensor table dùng face detection/alignment, Kinect và các perception algorithms; public không chứa face detection output. |
| **Tracking** | OpenPose/ROMP tìm nhiều người theo frame; paper không mô tả identity tracking bền vững qua clip. Multi-person association vì thế là rủi ro khi chọn đúng trẻ. | Kinect theo dõi một trẻ trong lượt đi; không mô tả tracker riêng. | Hệ thống realtime theo dõi child/objects và hợp nhất perception; thuật toán chi tiết nằm ở sensor-system references, không đóng gói đầy đủ trong public dataset. |
| **Pose estimation** | OpenPose (2D) và ROMP/SMPL (3D) từ một camera RGB. | Kinect v2, 25 joints. | Kinect-derived 3D upper-body skeleton; release giữ 10 joints + confidence. |
| **Gaze estimation** | Không. | Không. | Two-eye model–based gaze estimation từ Kinect/camera; public lưu heading vector 3D. |
| **Smoothing** | Không báo. | Không báo; audit local cho thấy raw có giá trị sentinel `100`, không nên tự smoothing mà không khai báo. | Không báo filter/smoothing trong dataset paper/schema. |
| **Normalization** | Không quy định normalization dataset-wide; tọa độ theo output OpenPose/ROMP. | Final table chứa feature nhiều đơn vị; không thấy normalization được phát hành. Raw workbook giữ tọa độ Kinect; pipeline cục bộ hiện không scale subject. | Tất cả modality hình học được đưa về **một hệ tọa độ Cartesian chung**; đây là coordinate registration, không đồng nghĩa statistical normalization. |
| **Missing data** | Clip nhiễu/occlusion/quality thấp bị loại, nhưng vẫn có pose failure/child ra khỏi khung ở dữ liệu giữ lại; không có policy imputation chính thức. | README nói không có missing values, nhưng audit cục bộ thấy **101 giá trị đúng bằng 100** tại 58 `(file, frame)` thuộc 3 ASD originals; ý nghĩa sentinel không được tài liệu hóa. | Schema cho phép `null` trong coordinate/confidence; paper không nêu imputation. Một số stream có thể thiếu/độ tin cậy thấp. |
| **Outlier** | Loại clip <3 s, lớp quá ít và clip chất lượng/lighting/occlusion không đạt; không nêu threshold định lượng. | Không nêu loại outlier. Nên flag sentinel, kiểm tra duplicate và không âm thầm sửa raw. | Face-expression output bị loại vì reliability thấp; không nêu rule outlier cho skeleton/gaze. |

## 14. Multimodal fusion

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Các modality** | Optical flow + 2D skeleton + 3D skeleton + demographic/clinical score. | RGB video + Kinect 3D skeleton/2D projection + engineered gait/joint features; classification table chủ yếu dùng feature vận động. | Upper-body skeleton + head orientation + eye-gaze + condition/task/ADOS/demographic metadata. |
| **Fusion level** | Dataset không áp một fusion model chính thức; người dùng có thể early/intermediate/late fuse. | `Final dataset.xlsx` là concatenation/feature-level table của 1.259 engineered features, nhưng không mô tả multimodal fusion RGB+skeleton. | Sensor system fusion/register các camera/Kinect vào common frame; release chỉ cung cấp aligned streams, không ép chiến lược ML fusion. |
| **Reliability** | OpenPose có confidence trong JSON; ROMP/flow reliability không được chuẩn hóa chung. Paper gợi ý tương lai dùng pose uncertainty/attention. | Không phát hành per-joint confidence; Kinect tracking state không rõ trong workbook. | Có `confidence` cho từng joint; paper loại facial expression vì reliability thấp. |
| **Missing modality** | Về kỹ thuật có thể dùng từng modality riêng; không có protocol chính thức cho missing-modality learning. | Có thể dùng skeleton/engineered table độc lập với RGB; không có policy chính thức. | Có thể dùng từng stream riêng và schema cho phép null, nhưng không có benchmark missing-modality chính thức. |

## 15. Cấu trúc dataset cuối

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Dataset hierarchy** | Participant → source session/video → activity clip → frame → person → joint. Clinical score nằm ở participant level. | Class (ASD/TD) → subject → original `video` hoặc `augmentation/1..7` → frame/joint; nhóm severe-ASD video-only tách riêng. | Participant folder → một JSON cho session/task segment → task metadata + frame-aligned arrays. Một intervention có thể tạo nhiều JSON parts. |
| **File structure** | Release đầy đủ theo modality/clip; workspace hiện chỉ có `ADOS_rating.xlsx`, README và sample repo, các thư mục modality chính đang trống. Repo sample có GIF/JPG/JSON/NPZ. | `Typical/{id}` và `Autism/children with ASD/{id}`; mỗi subject có `video/` và `augmentation/{1..7}/`; severe-ASD riêng. | `DREAMdataset/User {id}/*.json`; filename mã hóa user, segment/index, phase/intervention và timestamp. Workspace có 61 user folders, 3.121 JSON. |
| **Format** | Optical flow nguyên bản `.npy` (repo demo có bản `.jpg` grayscale); 2D JSON; 3D NPZ; clinical CSV/XLSX; raw RGB không public. | `.xlsx` 3D/2D/features, `.avi` RGB/skeleton/trajectory, `Final dataset.xlsx`, JPG consent, DOCX README. | JSON theo schema v1.1/v1.2; numeric arrays + metadata. |
| **ID system** | Filename gồm activity abbreviation, participant ID, condition/day/clip code; tài liệu không cung cấp formal grammar hoàn chỉnh. | Subject folder `1..50`; augmentation ID `1..7`; class nằm ở path. | Numeric participant ID; `$id`/filename và `date`; task có `index`, `ability`, `difficultyLevel`, `start`, `end`. |
| **Metadata** | Participant ID, DOB/chronological age, gender, ADOS module/SA/RRB/total/class/comparison score, motor-function level, severity; workbook local có chênh lệch roster. | Gender, age, walking length theo subject trong README; class/path, file-derived gait features. Không có clinical scale. | ID, age months, gender, ADOS pre/post fields, condition, task/difficulty/start/end, time, frame rate, schema version. |

## 16. Phân chia dữ liệu

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Train/Val/Test** | Không có split benchmark chính thức trong paper/repo. | Không có split chính thức. | Không có ML split chính thức; RET/SHT là randomized clinical condition, không phải train/test. |
| **Subject-independent** | Không được bảo đảm sẵn; khi tạo split phải group theo participant để tránh cùng trẻ ở train/test. | Không được bảo đảm. Tuyệt đối không random 800 file vì 7 augmentation cùng subject và còn có numeric duplicates chéo subject. | Không được bảo đảm; phải split theo `participant.id`, tốt nhất stratify condition/task. |
| **Cross-validation** | Không có LOSO/k-fold chính thức. | Không có official fold. Pipeline cục bộ có thể xây component-safe subject CV, nhưng đó là protocol mới chứ không phải protocol tác giả. | Không có official LOSO/k-fold. |
| **Class balance** | 11 lớp tương đối gần nhau nhưng không cân bằng tuyệt đối: 101–168 clip/lớp; drumming lớn nhất 168, squat nhỏ nhất 101. | Theo file: 400 A / 400 T; theo cohort: 50/50. Cân bằng số hàng không giải quyết duplicate/leakage. | Điều kiện gần cân bằng theo người (30 RET/31 SHT); task/session count không nhất thiết cân bằng, completion biến thiên. |

## 17. Quy mô dataset

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Participants** | Công bố 32 ASD; workbook local có 35 ID gốc nên cần dùng roster của clip release khi thực nghiệm. | 100 cohort chính (50 ASD, 50 TD) + 9 severe-ASD video-only chưa rõ overlap. | 61 public participants; 76 recruited, 70 randomized. |
| **Sessions** | Số session thực tế không báo; lịch dự kiến 4/tuần × 10 tuần. | Không có therapy session; 10 recording takes/người, 1 gait cycle giữ lại. | 3.121 public JSON/session segments; protocol 8 interventions + assessments/người nhưng completion khác nhau. |
| **Trials/samples** | **1.315 clips**, 11 activity classes. | **100 original selected gait cycles + 700 augmented = 800 samples**; thêm 10 severe-ASD videos ngoài bảng classification. | Không báo số discrete trials; **3.121 JSON** là session/segment units. |
| **Duration** | Hơn **108 h source recording**; phần clip public khoảng 1.315 × 7 s ≈ **2,56 h**. | Không báo tổng giờ; mỗi take đi 1,5–4 m/~2 cycles, một cycle giữ lại. | **306 h**. |
| **Frames/samples** | **244.679 frames**, trung bình 186,1 ± 92,9 frame/clip. | Raw skeleton audit: 21–60 observed frames/sample; không có tổng raw frame chính thức. Tensor cục bộ pad 60 không được xem là số frame quan sát. | Không báo exact frame count; 25 Hz × 306 h xấp xỉ 27,54 triệu time steps nếu toàn bộ 306 h được biểu diễn liên tục—chỉ là ước lượng. |

## 18. Privacy & Ethics

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Consent** | Có written parental consent trước enrollment; trẻ được thông báo quay. | Có file `Informed Consent Form for parents of children.jpg`; README nói phụ huynh đã ký. | Ít nhất một phụ huynh ký consent cho record và dùng khoa học ẩn danh. |
| **Ethics approval** | University of Delaware IRB **#637082-12**. | Không thấy số/ủy ban ethics trong README/paper kèm dữ liệu; consent không thay thế ethics approval. | Scientific Council of Babeș-Bolyai University, record **30664/10-02-2017**; trial NCT03323931. |
| **De-identification** | Không public raw RGB; phát hành optical flow/skeleton và clinical metadata. Tuy vậy movement/ID/clinical score vẫn là dữ liệu nhạy cảm. | Numeric case folders nhưng raw face-bearing RGB video được phát hành; mức de-identification thấp hơn hai bộ kia. | Loại mọi primary/direct measurement và biến có thể nhận dạng; dùng numeric ID, feature thứ cấp. |
| **Raw video availability** | Không; không truy cập khuôn mặt từ public release. | Có; màu video có thể thấy trẻ, gồm nhóm ASD nặng. | Không; raw RGB/audio/depth không public. |
| **Data access** | Công khai qua project/GitHub/linked Drive, nhưng workspace hiện không có toàn bộ modality. | Public download từ Dryad DOI. | Public feature dataset qua Swedish National Data Service; sample/schema/tool trên GitHub. |

## 19. License

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **License** | Repo có `LICENSE` Apache-2.0, còn arXiv paper hiển thị CC BY-NC-SA 4.0; **không thấy tuyên bố rõ rằng Apache-2.0 bao phủ toàn bộ data trên Drive**. Cần xin tác giả/xem terms của gói data trước khi phân phối/sử dụng thương mại. | Dryad phát hành dataset theo **CC0 1.0**. | Public secondary data: **CC BY-NC-SA 4.0**. |
| **Commercial use** | Chưa xác định cho dataset; Apache cho phép với code/repo, CC BY-NC-SA không cho commercial. Không nên suy rộng license code sang dữ liệu. | CC0 cho phép, nhưng quyền riêng tư/đạo đức đối với video trẻ em vẫn phải được tôn trọng. | Không cho commercial. |
| **Redistribution** | Chỉ nên redistribute phần có license rõ và giữ attribution/license; gói data cần xác minh riêng. | Được phép theo CC0; nên trích dẫn DOI/tác giả. | Được phép nếu attribution, non-commercial và share-alike; không bao gồm underlying raw recordings. |

## 20. Hạn chế dataset

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Technical limitations** | Video thực địa: illumination, clutter, resolution, pose variation, occlusion, người ra khỏi frame; pose detector có thể fail/đổi identity; thiếu subtle facial feature. Định dạng paper 17/24 joints khác release hiện tại 25/71. | Kinect noise/occlusion, rất ngắn, chỉ một gait cycle; global FPS/resolution/sync không tài liệu hóa; sentinel `100`; numeric/byte duplicates; engineered 1.259 features có nguy cơ overfit. | Chỉ 10 upper-body joints; derived estimate chứ không raw; null/confidence; face expression bị loại; thiếu raw audio/video/depth và trial outcome. |
| **Clinical limitations** | N=32, không TD, 5–12 tuổi, home intervention và ASD severity thiên moderate/high; không đủ cho chẩn đoán tổng quát. | Không công bố diagnostic criteria/clinical scale; cohort một địa điểm; tuổi hai nhóm khác nhau; giới/metadata có lỗi (README summary ghi 47 nam + 12 nữ cho cohort ASD 50). | 61 trẻ ASD 3–6 tuổi tại ba địa điểm Romania, không TD; 20 trẻ không hoàn tất full protocol; generalization theo văn hóa/ngôn ngữ/tuổi còn hạn chế. |
| **Annotation limitations** | Chỉ một nhãn activity/clip, không quality/correctness/engagement/gaze; không κ; 2–2 tie không mô tả. | Chỉ A/T, không behavior/time annotation; nhãn clinical provenance không mô tả. | Không public performance, correct/incorrect, engagement, emotion hay feedback/retry marker; do đó khó đánh giá response trực tiếp. |
| **Generalization** | Có tính tự nhiên nhưng camera/domain rất biến thiên; không có TD và sample nhỏ. Có thể dùng ngoài lab cho motion model nếu xử lý domain shift. | Chủ yếu áp dụng cho controlled straight-line gait; không suy rộng trực tiếp sang tương tác xã hội/trị liệu/tự nhiên. | Phù hợp therapy-table ABA/NAO; ngoài môi trường phòng trị liệu, độ tuổi 3–6 hoặc ngoài task JA/IM/TT cần validation mới. |

## 21. Khả năng tái tạo dataset

| Vấn đề | MMASD | DRYAD | DREAM |
|---|---|---|---|
| **Protocol availability** | Mức vừa: paper mô tả 10 tuần, tần suất, vai trò, 3 theme/11 activities và warm-up/debrief; chưa đủ lời thoại, số lần lặp, feedback và camera calibration để tái tạo y hệt. | Mức vừa-thấp: có đường đi, tốc độ, khoảng cách, 10 takes, camera height, nhiệt/ánh sáng; thiếu calibration, exact FPS, selection rule và chi tiết augmentation/feature implementation hoàn chỉnh. | Mức cao nhất trong ba bộ: paper, registry, 12 script/3 ability, discrete-trial flow, sensor setup, JSON schema, visualizer và source hệ thống. Exact raw data/clinical labels vẫn không public. |
| **Annotation reproducibility** | Có 11 định nghĩa lớp và majority vote, đủ annotate activity thô; chưa đủ tái tạo chính xác boundary/ambiguity/quality filters. | A/T dễ hiểu nhưng diagnostic ground truth không tái tạo được; engineered-feature definitions nhiều nhưng không phải tất cả implementation được đóng gói. | Task/condition/schema tái tạo được; không thể tự tái tạo response/engagement/emotion labels từ release vì ground truth không có. |
| **Equipment requirements** | Một RGB camera 25–30 FPS; trainer + adult model; robot/instruments/yoga props theo theme; OpenPose, ROMP, optical-flow processing; clinical assessor cho ADOS/SCQ/Vineland. | Kinect v2, máy tính Kinect SDK, Samsung Note 9 hoặc RGB camera tương đương, đường đi 1,5–4 m, kiểm soát ánh sáng/nhiệt; consent và clinical recruitment. | NAO, sensorized table/sandtray 26-inch, 3 RGB cameras, 2 Kinect, workstation đồng bộ, therapist + supervisor, ABA scripts và clinical ADOS expertise. |
| **Khả năng áp dụng cho dataset của mình** | Tái sử dụng taxonomy 11 activity, triadic child–trainer–model, privacy-first release (flow + 2D/3D pose), participant-level clinical table và multi-annotator majority vote. Nên bổ sung identity tracking, confidence và response-quality label. | Tái sử dụng controlled gait protocol, 25-joint 3D representation, engineered gait features và TD control. Bắt buộc split theo subject/duplicate component, không split augmentation theo hàng. | Tái sử dụng tốt nhất cho thiết kế robot therapy: stimulus–response–feedback–retry, ba skill/ba difficulty, RET-vs-SHT RCT, common-coordinate multimodal streams, task boundary và confidence. Nên thu thêm public-safe event labels và outcome từng trial ngay từ đầu. |

---

## Các chênh lệch phải ghi trong bài/luận văn

1. **MMASD N=32 nhưng workbook có 35 ID gốc.** Không được dùng workbook để tự đổi participant count của benchmark; cần map clip → participant rồi chốt roster.
2. **MMASD skeleton version mismatch.** Paper v3 nói 17 joints 2D + 24 joints 3D; README/release hiện tại và sample thực tế là 25 + 71. Luôn ghi commit/version và shape thực tế.
3. **DRYAD 700 vs 800.** Abstract nói 700 augmentation folders (350+350); README/`Final dataset.xlsx` và folder audit cho 800 instances khi cộng 100 originals. Cách viết an toàn: 100 originals + 700 augmentations = 800 samples.
4. **DRYAD duplicate/sentinel.** Cân bằng 400/400 không bảo đảm đánh giá sạch; split ngẫu nhiên theo hàng làm leakage. Giá trị `100` chưa rõ nghĩa phải flag, không tự coi là tọa độ hợp lệ hay tự impute.
5. **DREAM 76/70/61.** Đây là các giai đoạn khác nhau: recruited / randomized / retained-public. Bản cục bộ và paper dataset dùng 61, 3.121 JSON, 306 h.
6. **“Đã thu” khác “được phát hành”.** DREAM thu RGB/RGB-D/audio và nhiều perceptions nhưng public chỉ là secondary data; MMASD quay RGB nhưng không public raw; DRYAD lại phát hành RGB có thể nhận dạng.

## Gợi ý lựa chọn cho đề tài

- Nếu mục tiêu là **phân loại động tác hoặc đánh giá vận động trong trị liệu**: bắt đầu từ MMASD.
- Nếu mục tiêu là **ASD/TD từ gait/skeleton 3D**: dùng DRYAD, với subject/duplicate-safe split.
- Nếu mục tiêu là **joint attention, imitation, turn-taking hoặc robot-assisted therapy**: DREAM phù hợp nhất về protocol, nhưng cần tự xây dựng nhãn response/correctness/engagement nếu nghiên cứu cần các outcome đó.
- Nếu muốn gộp cả ba, chỉ nên chuẩn hóa ở tầng **skeleton/time series**; không thể coi taxonomy label là đồng nhất. MMASD là activity, DRYAD là diagnosis group, DREAM là therapy task/condition.

## Nguồn chính

### MMASD

- Paper HTML: <https://arxiv.org/html/2306.08243>
- DOI: <https://doi.org/10.1145/3577190.3614117>
- Repo: <https://github.com/Li-Jicheng/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis>
- Local: `MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis/README.md`, `data_MMASD/ADOS_rating.xlsx`, sample JSON/NPZ.

### DRYAD

- Dataset DOI: <https://doi.org/10.5061/dryad.s7h44j150>
- Local: `Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset/readme.docx`, `Final dataset.xlsx`, raw folders.
- Audit/contract cục bộ: `DATA_DRYAD_DESCRIPTION.md`, `artifacts/dryad_msg3d/DATA_CONTRACT_v1.md`.
- Dryad reuse/CC0: <https://v3.datadryad.org/help/guides/reuse>

### DREAM

- Paper: <https://doi.org/10.1371/journal.pone.0236939>
- Data DOI: <https://doi.org/10.5878/17p8-6k13>
- Repo/schema: <https://github.com/dream2020/data>
- Trial: <https://clinicaltrials.gov/study/NCT03323931>
- Local: `data_Dream/documentation/DREAM-data.pdf`, `data_Dream/dream.1.2.json`, `data_Dream/DREAMdataset/`.
