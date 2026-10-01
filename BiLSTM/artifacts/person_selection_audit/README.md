# Audit person selection / child tracking — Protocol A

Ngày audit: 2026-09-14. Phạm vi: 1.374 clip usable, 10 activity, 254.674 OpenPose JSON frame từ `artifacts/paper_repro/manifest.csv`.

Không train model; không sửa source/config/data; visualization chỉ dùng skeleton-coordinate canvas, không overlay RGB. Hash đầu vào nằm trong `input_hashes.sha256`.

## Quy ước bằng chứng

- **Observed fact**: đo trực tiếp từ code/JSON/artifact local.
- **Heuristic proxy**: tín hiệu hình học hoặc raw array index; không phải true person identity.
- **Inference**: diễn giải có giới hạn từ observed fact + proxy.

OpenPose không cung cấp persistent ID: scan toàn bộ tìm thấy 901.570 person object và tất cả đều có `person_id=[-1]`. Vì thế `people[i]` chỉ là raw candidate index của frame, tuyệt đối không được đọc là track ID.

## 1. Trace implementation hiện tại

1. `src/data/discover.py:36-42`: các JSON frame được sort theo số trong hậu tố `_<index>_keypoints.json`.
2. `src/data/openpose_parser.py:47-73`: đọc JSON object, lấy `payload["people"]`; mỗi `pose_keypoints_2d` phải có đúng 75 giá trị và được reshape thành `[25,3] = (x,y,confidence)`. Parser giữ nguyên thứ tự raw `people[]`.
3. `src/data/child_tracking.py:39-69`: một joint hợp lệ khi coordinate/confidence hữu hạn, `confidence >= 0.1`, và `(x,y)!=(0,0)`. Candidate chỉ cần **ít nhất một joint hợp lệ**; không có minimum joint count/torso-anchor requirement. Feature:
   - bbox `low=min(points)`, `high=max(points)`;
   - `center=(low+high)/2`;
   - `diagonal=max(||high-low||_2,1)`;
   - `torso=mean` của các joint hợp lệ trong MidHip, Neck, RHip, LHip, RShoulder, LShoulder; nếu không có thì dùng bbox center;
   - `mean_confidence=mean(confidence của joint hợp lệ)`, `valid_count=#joint hợp lệ`.
4. `src/data/child_tracking.py:72-79`: khi chưa có previous candidate, score lớn hơn tốt hơn:

   `initial_score = mean_confidence * ln(1 + valid_count) / sqrt(diagonal)`

   do Protocol A có `prefer_smaller_person=true`. Size không có một linear weight; nó vào mẫu số với exponent `-1/2`. Nếu tắt prior thì score chỉ còn `confidence * ln(1+valid_count)`.
5. `src/data/child_tracking.py:116-139`: nếu frame không có candidate hợp lệ, chọn index `-1`, person `None`, log `no_valid_person`; biến `previous` không reset. First valid frame chọn max initial score, tie-break index nhỏ hơn. Các frame sau với `temporal_nearest` chọn candidate có transition cost nhỏ nhất:

   `0.45 * center_distance/scale + 0.35 * torso_distance/scale + 0.15 * |ln(size_t/size_prev)| + 0.05 * (1-confidence_t)`

   với `scale=max(diagonal_prev, diagonal_t, 1)`. Không có distance gate, maximum-cost rejection, global track optimization, velocity model, hay persistent assignment.
6. `src/data/dataset.py:111-153`: dataset parse cả clip, gọi tracker bằng threshold/config trên, rồi chuyển `selected_people` sang preprocessing.
7. `src/data/preprocessing.py:147-195`: frame `None` thành zero raw frame; joint thiếu được interpolate theo config, skeleton center tại MidHip, scale bằng shoulder width hoặc torso fallback, rồi xuất `[T,25,2]`.
8. `src/data/dataset.py:153-160`: Protocol A uniform-resample final normalized sequence về 60 frame. Do đó final model input không còn candidate khác và không giữ confidence/person index.

`selected_score_mean` trong inspection table được định nghĩa là âm của mean transition cost (bỏ first acquisition), nên gần 0/lớn hơn là tốt hơn. `selected_initial_score` được lưu riêng vì initial score và transition cost khác đơn vị, không nên trộn thành một mean.

## 2. Candidate statistics

Raw `people[]` và valid-candidate counts giống hệt nhau ở dataset này: mọi person object đều có ít nhất một joint qua threshold 0,1. Nhưng điều đó không có nghĩa mọi candidate là full person.

| Activity | Frames | 0 people | 1 person | 2 people | >=3 people |
|---|---:|---:|---:|---:|---:|
| ALL | 254674 | 0 (0.000%) | 89 (0.035%) | 7101 (2.788%) | 247484 (97.177%) |
| arm_swing_as | 16650 | 0 (0.000%) | 1 (0.006%) | 429 (2.577%) | 16220 (97.417%) |
| body_swing_bs | 24710 | 0 (0.000%) | 1 (0.004%) | 71 (0.287%) | 24638 (99.709%) |
| chest_expansion_ce | 20615 | 0 (0.000%) | 0 (0.000%) | 77 (0.374%) | 20538 (99.626%) |
| drumming_dr | 64524 | 0 (0.000%) | 87 (0.135%) | 4837 (7.496%) | 59600 (92.369%) |
| maracas_forward_shaking_mfs | 19645 | 0 (0.000%) | 0 (0.000%) | 38 (0.193%) | 19607 (99.807%) |
| maracas_shaking_ms | 24825 | 0 (0.000%) | 0 (0.000%) | 254 (1.023%) | 24571 (98.977%) |
| sing_and_clap_sac | 24205 | 0 (0.000%) | 0 (0.000%) | 139 (0.574%) | 24066 (99.426%) |
| squat_sq | 20110 | 0 (0.000%) | 0 (0.000%) | 200 (0.995%) | 19910 (99.005%) |
| tree_pose_tr | 20015 | 0 (0.000%) | 0 (0.000%) | 1043 (5.211%) | 18972 (94.789%) |
| twist_pose_tw | 19375 | 0 (0.000%) | 0 (0.000%) | 13 (0.067%) | 19362 (99.933%) |

Observed exact candidate range là 1–9, mean 3,540/frame. Thuật toán phải chọn giữa nhiều candidate ở **254.585/254.674 frame = 99,965%**.

## 3. Continuity metrics và identity-switch proxy

Mỗi clip có: frame counts; raw-index changes; normalized center displacement; absolute log-size jump; confidence jump; missing selection; selected confidence/score; anomaly frame indices.

Primary thresholds được lấy ở Q99,5 của 211.597 transition có selected raw index không đổi, rồi lấy max với Q99,5 của 75 transition one-candidate→one-candidate. Đây chỉ là lower-discontinuity reference, không phải identity ground truth:

- centroid displacement normalized > **0,163691**;
- absolute log size ratio > **0,343866**, tương đương size ratio > **1,4104×** hoặc < 1/1,4104;
- absolute confidence jump > **0,128544** (được báo cáo, không dùng để gán identity proxy).

Một `identity_switch_proxy` event cần ở cạnh frame có >=2 valid candidates và đồng thời có ít nhất hai trong ba tín hiệu: centroid anomaly, scale anomaly, raw selected-index change. Đây là proxy bảo thủ cho candidate-assignment discontinuity; nó vẫn có thể bắt detection fragmentation/occlusion hoặc OpenPose reordering thay vì true identity switch.

Observed aggregate:

- 0 missing-selected-person frame; không có clip no-person.
- 41.703 raw selected-index changes trong 253.300 adjacent transitions (16,464%); 1.289/1.374 clip có ít nhất một change. Đây **không phải** true identity switch.
- 1.489 centroid-jump events trong 256 clip; 1.521 scale-jump events trong 253 clip; 1.635 confidence-jump events trong 365 clip.
- 1.390 combined identity-switch-proxy events trong 253 clip = **18,413% clip**.
- Sensitivity theo stable-index reference: Q99 → 316 clip (22,999%); Q99,5 → 253 (18,413%); Q99,9 → 139 (10,116%).

## 4. Tracking risk

Rule không nhằm chia ba nhóm cân bằng:

- **HIGH**: có ít nhất một combined identity-switch proxy; hoặc no-person ratio >=5%. Branch missing không kích hoạt ở dataset này.
- **MEDIUM**: không HIGH nhưng có centroid/scale anomaly, missing selection, hoặc raw-index change trong multi-person clip.
- **LOW**: không có các evidence trên.

Kết quả: LOW 78 (5,677%), MEDIUM 1.043 (75,910%), HIGH 253 (18,413%). Đây là inspection priority, không phải ground-truth error labels.

## 5. Audit “smaller skeleton = child”

Observed size là diagonal của axis-aligned bbox trên mọi joint hữu hạn/nonzero có confidence >=0,1.

- Size prior chỉ tác động trực tiếp ở first valid acquisition; temporal frames sau không tiếp tục thêm `1/sqrt(size)` mà dùng 15% absolute log-size-change trong transition cost.
- Trong 254.585 multi-candidate frames, selected candidate là exact smallest ở 119.054 frame = **46,764%**. Median `selected / smallest = 1,043`; Q75=1,642; Q95=3,884.
- So với runner-up dưới objective đang active, median selected-size ratio là **0,781** (Q25=0,675; Q75=0,934; Q95=1,484).
- Ở 1.374 initial acquisitions, smallest candidate được chọn 53,275%. Size prior đổi lựa chọn so với quality-only score ở **499 clip = 36,317%**.
- Initial candidate có <10 valid joints ở 35 clip (2,547%), <5 joint ở 20 clip (1,456%), diagonal <50 px ở 31 clip (2,256%). Visuals cho thấy một số candidate này là fragment, không phải full skeleton.

Inference: tracker không phải “luôn chọn người nhỏ nhất” xuyên suốt clip, nhưng size prior có ảnh hưởng vật chất ở điểm khởi tạo và có failure mode rõ ràng: candidate fragment nhỏ có thể thắng acquisition rồi tracker không có gate/recovery để từ chối hoặc tạo track mới.

## 6. Visual audit

Đã tạo 30 contact sheet: 10 LOW, 10 MEDIUM, 10 HIGH, greedy-cover activity/subject/session; mỗi clip có 8 frame gồm anomaly ưu tiên và uniform coverage. Orange là selected; blue là candidate khác; label có raw index, confidence, bbox diagonal, objective value. Không dùng RGB vì local `data_MMASD` chỉ có 479.854 JPG dưới `optical_flow`; không có full original RGB frames/video joinable.

Visual observation không cần role inference:

- `mfs_40504_D16_001_n`: selected fragment diagonal≈33 ở t=21–22 rồi nhảy sang full skeleton diagonal≈352 tại t=23.
- `mfs_40504_D16_004_y`: initial selected candidate chỉ có 2 valid joints/diagonal≈32 rồi nhảy sang full skeleton ở t=1.
- `ms_40683_D8_002_i`: tracker nhiều lần bám candidate fragment nhỏ bên phải trong khi nhiều full-body candidates tồn tại.
- `ce_40493_D16_004_n`: tracker chọn upper-body/fragment-like detection trong nhiều sampled frames dù có full-body candidates.
- Một số HIGH khác chỉ đổi raw index nhưng giữ vùng/hình học tương đối liên tục; vì vậy không đồng nhất HIGH với true identity switch.

## 7. Top 20 suspicious clips

`top20_suspicious_clips.csv` xếp hạng theo combined-proxy count, normalized maximum jump severities và raw-index-change term; mỗi row có reason và toàn bộ frame indices. `top20_windows/` có cửa sổ t-2…t+2 quanh event severity cao nhất cho từng clip.

Top 5: `mfs_40504_D16_001_n` (39 proxy events), `mfs_40683_D16_001_i` (32), `dr_20613_D8_005_i` (26), `ce_40493_D16_004_n` (25), `sq_40074_D1_005_n` (24).

## 8. Child identity / metadata và MMASD+

- Exhaustive local OpenPose scan: 901.570/901.570 `person_id` fields bằng `[-1]`.
- Prepared 120-clip manual identity review hiện có 0/120 human labels và 0 person-role pairs; không có manual ground truth để tính true wrong-person/switch rate.
- MMASD+ README dòng 33 nói individual persons được isolate bằng YOLOv8 + Deep SORT. Checked-in `enhanced-mmasd/Code/*.py` không chứa implementation role assignment hay exported per-frame track table.
- Local MMASD+ inventory có 3.276 MediaPipe CSV, 3.267 MP4, 3.251 ROMP output directories và 642.636 NPZ frames; không có explicit YOLO/DeepSORT/bbox export. Filename match có 1.069/1.374 frozen clips, nhưng 0 matched CSV có equal OpenPose frame count; không có timestamp/crop transform/raw-person-index mapping.
- `ASD_Label` không được chứng minh là child/therapist role (có cả values 0,1,2,3,6). ROMP geometry chỉ là anonymous track-like evidence. Vì thế các artifact này chưa thể validate hay thay current selection một cách an toàn.

## 9. Per-activity tracking risk

`mean_jump_score` là mean per-clip của `max(center_jump/center_threshold, scale_jump/scale_threshold)` trên transitions; nhỏ hơn tốt hơn.

| Activity | Clips | Multi-person frames | HIGH | Mean jump score | Mean selected confidence |
|---|---:|---:|---:|---:|---:|
| arm_swing_as | 105 | 99.994% | 19 (18.10%) | 0.0955 | 0.7034 |
| body_swing_bs | 117 | 99.996% | 27 (23.08%) | 0.0895 | 0.7075 |
| chest_expansion_ce | 114 | 100.000% | 21 (18.42%) | 0.0855 | 0.7076 |
| drumming_dr | 343 | 99.865% | 75 (21.87%) | 0.1177 | 0.6763 |
| maracas_forward_shaking_mfs | 101 | 100.000% | 20 (19.80%) | 0.0784 | 0.7047 |
| maracas_shaking_ms | 130 | 100.000% | 19 (14.62%) | 0.0739 | 0.7125 |
| sing_and_clap_sac | 113 | 100.000% | 34 (30.09%) | 0.1208 | 0.6508 |
| squat_sq | 102 | 100.000% | 26 (25.49%) | 0.0905 | 0.7112 |
| tree_pose_tr | 129 | 100.000% | 7 (5.43%) | 0.0451 | 0.7438 |
| twist_pose_tw | 120 | 100.000% | 5 (4.17%) | 0.0395 | 0.6556 |

`sing_and_clap_sac` khó nhất theo HIGH rate và mean jump score. `drumming_dr` đứng thứ hai về mean jump score, có 75 HIGH clips (nhiều nhất theo count) và confidence tương đối thấp. MFS có visible fragment failures dù aggregate HIGH rate thấp hơn SAC/Squat/Body Swing. MS thấp hơn theo proxy rate nhưng một visual HIGH cho thấy persistent fragment selection; class aggregate không thay thế clip-level review.

## 10. Kết luận

**OUTCOME C — TRACKING IS A MAJOR PROBLEM**, với qualification: evidence đủ mạnh cho candidate-selection/tracking failure, nhưng không đủ để gọi selected candidate là therapist hay để đo true child-identity error. Lý do: selection thực sự multi-person ở 99,965% frame; tracker chỉ greedy one-step, không gate; 10,1–23,0% clip có conservative continuity proxy qua sensitivity range; và skeleton-only visual xác nhận các failure fragment→full-person / persistent-fragment rõ ràng. Không train nên chưa định lượng phần accuracy gap do tracking gây ra.

Sáu câu trả lời:

1. Frame >1 person: **99,965%**.
2. Clip có dấu hiệu identity-switch proxy: **18,413% primary**; sensitivity **10,116–22,999%**. Đây không phải true identity-switch rate.
3. Activity risk cao nhất: **Sing and Clap** (30,09% HIGH; mean jump 0,1208); Drumming có 75 HIGH clip và mean jump 0,1177.
4. Bias nhỏ nhất: không phải luôn chọn smallest (46,764% multi frames), nhưng bias initialization là **material**: thay quality-only choice ở 36,317% clip và đôi lúc chọn tiny fragment.
5. Evidence chọn therapist: **không thể xác nhận** vì thiếu RGB/role ground truth/persistent ID. Có evidence rõ cho wrong/fragment candidate selection và spatial assignment discontinuity, không có evidence role-specific therapist.
6. Có cần sửa trước sequence/window audit: **Có**. Chỉ đề xuất: reject fragment candidates bằng minimum key-joint/torso completeness; tạo tracklets toàn clip với assignment + motion/scale gating và explicit lost/reacquisition; score child ở track level thay vì first-frame size prior; sau đó human-review/validate một subset có RGB hoặc role mapping đáng tin cậy trước khi dùng tracker mới. Không dùng MMASD+ label/token làm child ground truth nếu chưa align và xác minh.

## Artifacts

- `clip_tracking_inspection.csv`: 1 row/clip, đầy đủ metrics/risk/event indices.
- `frame_person_counts.csv`: raw và valid counts overall/per activity.
- `multi_person_size_frames.csv`: 1 row/multi-candidate frame, selected/runner-up size và ratio.
- `per_activity_tracking_risk.csv`: bảng class risk.
- `thresholds_and_rules.json`, `threshold_sensitivity.csv`, `size_bias_summary.json`, `summary.json`.
- `visual_sample_manifest.csv`, `visual_samples/{low,medium,high}/`.
- `top20_suspicious_clips.csv`, `top20_visual_manifest.csv`, `top20_windows/`.
