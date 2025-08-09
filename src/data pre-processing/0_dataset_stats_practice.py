import pandas as pd
from datetime import timedelta

# Load your data
csv_path = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. VIOLENCEDETECTIONPROJECT\annotations\Final\preprocessed_dataset_1.csv"
df = pd.read_csv(csv_path, encoding='latin1')

# Duration
df['Duration'] = df['End time(s)'] - df['Start time(s)']
total_segments = len(df)
total_videos = df['Video ID'].nunique()
total_duration_sec = df['Duration'].sum()
total_duration_hms = str(timedelta(seconds=int(total_duration_sec)))
avg_duration = df['Duration'].mean()

# Summary DataFrame
summary_df = pd.DataFrame({
    'Metric': [
        'Total annotated segments',
        'Total unique videos',
        'Total duration (seconds)',
        'Total duration (HH:MM:SS)',
        'Average duration per segment (sec)'
    ],
    'Value': [
        total_segments,
        total_videos,
        int(total_duration_sec),
        total_duration_hms,
        round(avg_duration, 2)
    ]
})

video_durations = df.groupby('Video ID')['End time(s)'].max()

avg_video_duration = round(video_durations.mean(), 2)
min_video_duration = round(video_durations.min(), 2)
max_video_duration = round(video_durations.max(), 2)
std_video_duration = round(video_durations.std(), 2)

video_stats_df = pd.DataFrame({
    'Metric': [
        'Average video duration (sec)',
        'Minimum video duration (sec)',
        'Maximum video duration (sec)',
        'Std deviation of video durations (sec)'
    ],
    'Value': [
        avg_video_duration,
        min_video_duration,
        max_video_duration,
        std_video_duration
    ]
})

# Append video-level stats to summary
summary_df = pd.concat([summary_df, video_stats_df], ignore_index=True)

# Violence Type Count
violence_counts = df['Violence Type (Video)'].value_counts().reset_index()
violence_counts.columns = ['Violence Type (Video)', 'Count']

# Video Type Count
video_type_counts = df['Video Type'].value_counts().reset_index()
video_type_counts.columns = ['Video Type', 'Count']

# Save to Excel with multiple sheets
output_path = csv_path.replace('.csv', '_FULL_STATS.xlsx')
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    violence_counts.to_excel(writer, sheet_name='Violence Types', index=False)
    video_type_counts.to_excel(writer, sheet_name='Video Types', index=False)

print("✅ Stats saved to:", output_path)
