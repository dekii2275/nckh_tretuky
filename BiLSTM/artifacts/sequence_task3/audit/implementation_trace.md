# Task 3 — current sequence implementation trace

The source path is parsed into all raw OpenPose frames, then Tracker V2 selects one full-clip person track, then missing-joint handling and hip/shoulder normalization run on the full track, and only then is a 60-step sequence selected.

1. Raw JSON clip parsing: `src/data/openpose_parser.py:76`.
2. Tracker dispatch/person selection: `src/data/tracking.py:13`.
3. Missing-joint preprocessing and normalization: `src/data/preprocessing.py:130`.
4. Dataset ordering (parse → track → preprocess → extract): `src/data/dataset.py:78`.
5. Central sequence construction: `src/data/sequence.py:78`; exact resampling: `src/data/sequence.py:46`.
6. Batch collation pads (all Task-3 samples are already 60 frames) and returns `[B,60,25,2]`; each item is `[60,25,2]`.

Current `full_clip_resample` semantics:

- Resampling occurs after both tracking and normalization.
- For N > 60: `positions=linspace(0,N-1,60)`, indices use NumPy `rint` (round-to-nearest, ties-to-even), endpoints included.
- For N = 60: an exact copy is returned.
- For N < 60: linear temporal interpolation between adjacent normalized poses produces 60 outputs; clips are never silently dropped.
- For N > 60, selected indices cannot duplicate; for N < 60, interpolation positions reuse source endpoints/neighbors but output poses are interpolated rather than duplicated (except N=1).
