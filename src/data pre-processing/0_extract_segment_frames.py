import os
import cv2
import pandas as pd

# Paths
BASE_PATH = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. VIOLENCEDETECTIONPROJECT"
VIDEO_DIR = os.path.join(BASE_PATH, "DATASET", "youtube_videos")
ANNOTATION_PATH = os.path.join(BASE_PATH, "annotations", "Final", "preprocessed_dataset_1.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "DATASET", "00. Actual Dataset", "Frames")

# Load annotation CSV
df = pd.read_csv(ANNOTATION_PATH, encoding='latin1')

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through each segment
for idx, row in df.iterrows():
    video_id = str(row['Video ID'])
    segment_id = str(row['Segment ID'])
    start_frame = int(row['Start frame'])
    end_frame = int(row['End frame'])

    # Prepare segment folder
    segment_folder = os.path.join(OUTPUT_DIR, segment_id)
    if os.path.exists(segment_folder) and len(os.listdir(segment_folder)) > 0:
        print(f"[SKIP] Segment {segment_id} already exists.")
        continue

    os.makedirs(segment_folder, exist_ok=True)

    # Match any video file that starts with the video_id
    matched_file = None
    for file in os.listdir(VIDEO_DIR):
        if file.startswith(f"{video_id}_"):
            matched_file = os.path.join(VIDEO_DIR, file)
            break

    if not matched_file:
        print(f"[ERROR] Video {video_id} not found.")
        continue

    # Open video and extract frames
    cap = cv2.VideoCapture(matched_file)
    frame_num = 0
    saved_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_num > end_frame:
            break
        if frame_num >= start_frame:
            filename = f"frame_{saved_count + 1:05d}.jpg"
            filepath = os.path.join(segment_folder, filename)
            success_save = cv2.imwrite(filepath, frame)
            if not success_save:
                print(f"[ERROR] Failed to save frame {saved_count + 1} at {filepath}")
            else:
                saved_count += 1
        frame_num += 1

    cap.release()
    print(f"[DONE] Segment {segment_id} → {saved_count} frames saved.")


# Okay, I've stopped at some points but for example, in 166_1, then there are 100 frames(for example), if I've stopped at 80
# then this code just skips the 166_1, not 166_2. 
# So what I need is check how many frames it should be in segment ID, and matches with number of image files in each folder
# then re run the code only to get the frame images that are missing.