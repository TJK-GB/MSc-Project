import os
import pandas as pd

# --- Paths ---
BASE_PATH = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. VIOLENCEDETECTIONPROJECT"
VIDEO_DIR = os.path.join(BASE_PATH, "DATASET", "youtube_videos")
ANNOTATION_PATH = os.path.join(BASE_PATH, "annotations", "Final", "preprocessed_dataset_1.csv")
FRAMES_BASE_PATH = os.path.join(BASE_PATH, "DATASET", "00. Actual Dataset", "Frames")  # e.g., ...\Frames\166_1

# --- Load Annotations ---
df = pd.read_csv(ANNOTATION_PATH, encoding = 'latin1')

# --- Check Each Segment Folder ---
mismatches = []

for _, row in df.iterrows():
    video_id = str(row['Video ID'])
    segment_id = str(row['Segment ID'])
    start = int(row['Start frame'])
    end = int(row['End frame'])

    expected_count = end - start + 1
    segment_folder = os.path.join(FRAMES_BASE_PATH, f"{segment_id}")

    if not os.path.exists(segment_folder):
        mismatches.append((f"{segment_id}", "Folder Missing", expected_count, 0))
        continue

    jpg_files = [f for f in os.listdir(segment_folder) if f.endswith(".jpg")]
    actual_count = len(jpg_files)

    if actual_count != expected_count:
        mismatches.append((f"{segment_id}", "Count Mismatch", expected_count, actual_count))

# --- Print Mismatches ---
for segment, reason, expected, actual in mismatches:
    print(f"{segment}: {reason} — Expected: {expected}, Found: {actual}")
