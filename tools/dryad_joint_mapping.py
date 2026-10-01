"""Explicit Dryad-to-NTU joint mapping used for diagnostics only.

The raw names come from Dryad headers.  The model names/order are the
standard NTU-25 semantic convention inferred from the numeric edge list in
``autism/graph/ntu_rgb_d.py``; the released graph file does not name joints.
"""

RAW_JOINTS = (
    "Midspain", "AnkleLeft", "AnkleRight", "ElbowLeft", "ElbowRight",
    "FootLeft", "FootRight", "HandLeft", "HandRight", "HandTipLeft",
    "HandTipRight", "Head", "HipLeft", "HipRight", "KneeLeft",
    "KneeRight", "Neck", "ShoulderLeft", "ShoulderRight", "SpineBase",
    "SpineShoulder", "ThumbLeft", "ThumbRight", "WristLeft", "WristRight",
)

MODEL_JOINTS = (
    "SpineBase", "SpineMid", "Neck", "Head", "ShoulderLeft",
    "ElbowLeft", "WristLeft", "HandLeft", "ShoulderRight", "ElbowRight",
    "WristRight", "HandRight", "HipLeft", "KneeLeft", "AnkleLeft",
    "FootLeft", "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
)

# Zero-based raw index -> zero-based model index.
RAW_TO_MODEL = (
    1, 14, 18, 5, 9, 15, 19, 7, 11, 21,
    23, 3, 12, 16, 13, 17, 2, 4, 8, 0,
    20, 22, 24, 6, 10,
)

MODEL_TO_RAW = tuple(RAW_TO_MODEL.index(model_index) for model_index in range(25))
