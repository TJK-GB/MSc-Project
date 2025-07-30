import pandas as pd
import os
import math

# === CONFIGURATION ===
ANNOTATION_FILE = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\annotations\Original\Annotation_test\annotations_real(time_level)_ver1.xlsx"
OUTPUT_FILE = ANNOTATION_FILE.replace(".xlsx", "_with_frames.xlsx")
FPS = 30  # Adjust if you know the exact FPS per video

# === TIME CONVERSION FUNCTION ===
def time_to_seconds(t):
    if isinstance(t, str):
        parts = t.strip().split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        elif len(parts) == 1:
            return float(parts[0])
    return float(t)

# === PROCESSING ===
df = pd.read_excel(ANNOTATION_FILE)

# Convert start and end times to seconds
df["start_seconds"] = df["Video Start"].apply(time_to_seconds)
df["end_seconds"] = df["Video End"].apply(time_to_seconds)

# Convert seconds to frames
df["start_frame"] = df["start_seconds"].apply(lambda x: math.ceil(x * FPS))
df["end_frame"] = df["end_seconds"].apply(lambda x: math.floor(x * FPS))

# Save updated file
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Done. Saved with frame columns to:\n{OUTPUT_FILE}")
