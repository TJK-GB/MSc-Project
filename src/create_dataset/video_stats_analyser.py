import yt_dlp

def get_video_durations(video_urls):
    durations = []
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in video_urls:
            try:
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                durations.append(duration)
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")

    return durations

def print_stats(durations):
    total_videos = len(durations)
    total_duration = sum(durations)
    avg_duration = total_duration / total_videos if total_videos else 0

    print(f"\n Video Stats")
    print(f"Total videos: {total_videos}")
    print(f"Total duration: {total_duration / 60:.2f} minutes")
    print(f"Average duration: {avg_duration:.2f} seconds ({int(avg_duration // 60)}m {int(avg_duration % 60)}s)")

def read_links_from_file(filepath):
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    file_path = "youtube_links.txt"  # Or change to your actual file
    urls = read_links_from_file(file_path)
    durations = get_video_durations(urls)
    print_stats(durations)
