# youtube_annotator_core.py (updated to use downloaded videos instead of YouTube links)

import tkinter as tk
from tkinter import ttk
import os
import pandas as pd
import cv2
# from PIL import Image, ImageTk
import re

ANNOTATION_FILE = "annotations_real.xlsx"
DOWNLOAD_DIR = "./youtube_videos"

class YouTubeAnnotationApp:
    def __init__(self, master, video_files):
        self.master = master
        self.video_files = video_files
        self.current_index = 0
        self.annotations = []

        # Declare variables BEFORE using them
        self.violence_type_var = tk.StringVar()
        self.video_type_var = tk.StringVar()
        self.sound_violence_var = tk.StringVar()
        self.sound_type_var = tk.StringVar()
        self.text_violence_var = tk.StringVar()
        self.text_content_var = tk.StringVar()
        self.text_content_var_1 = tk.StringVar()
        self.video_start_var = tk.StringVar()
        self.video_end_var = tk.StringVar()

        self.master.title("Video Annotation Tool")
        self.video_label = tk.Label(master, text=f"Video {self.current_index + 1} / {len(self.video_files)}")
        self.video_label.grid(row=0, column=0, columnspan=3, pady=(10, 5))

        form_frame = tk.Frame(master)
        form_frame.grid(row=1, column=0, columnspan=3, padx=20)

        # Row 0 - Dropdowns 1
        tk.Label(form_frame, text="Violence Type(Video)").grid(row=0, column=0, padx=10, pady=5)
        tk.Label(form_frame, text="Violence Type(Sound)").grid(row=0, column=1, padx=10, pady=5)
        tk.Label(form_frame, text="Violence Type(Text)").grid(row=0, column=2, padx=10, pady=5)

        ttk.Combobox(form_frame, textvariable=self.violence_type_var, values=[" ", "Violent", "Non-violent", "N/A", "Others"], state="readonly").grid(row=1, column=0, padx=10, pady=2)
        ttk.Combobox(form_frame, textvariable=self.sound_violence_var, values=[" ", "Violent", "Non-violent", "N/A", "Others"], state="readonly").grid(row=1, column=1, padx=10, pady=2)
        ttk.Combobox(form_frame, textvariable=self.text_violence_var, values=[" ", "Violent", "Non-violent", "N/A", "Others"], state="readonly").grid(row=1, column=2, padx=10, pady=2)

        # Row 2 - Dropdowns 2
        tk.Label(form_frame, text="Video Type").grid(row=2, column=0, padx=10, pady=5)
        tk.Label(form_frame, text="Sound Type").grid(row=2, column=1, padx=10, pady=5)
        tk.Label(form_frame, text="Text Content").grid(row=2, column=2, padx=10, pady=5)

        ttk.Combobox(form_frame, textvariable=self.video_type_var, values=[" ", "News", "CCTV", "Self-filmed", "Dashcam", "Bodycam", "Others"], state="readonly").grid(row=3, column=0, padx=10, pady=2)
        ttk.Combobox(form_frame, textvariable=self.sound_type_var, values=[" ", "Gunshot", "Screaming", "Crying", "Shouting", "Others"], state="readonly").grid(row=3, column=1, padx=10, pady=2)
        tk.Entry(form_frame, textvariable=self.text_content_var, width=20).grid(row=3, column=2, padx=10, pady=2)

        # Row 3 - Time & Buttons
        tk.Label(form_frame, text="Start time").grid(row=5, column=1, padx=10, pady=2)
        tk.Entry(form_frame, textvariable=self.video_start_var, width=20).grid(row=6, column=1, padx=10, pady=2)

        tk.Button(form_frame, text="Play Video", command=self.play_video).grid(row=6, column=0, padx=10, pady=5)


        tk.Label(form_frame, text="End time").grid(row=5, column=2, padx=10, pady=2)
        tk.Entry(form_frame, textvariable=self.video_end_var, width=20).grid(row=6, column=2, padx=10, pady=2)

        tk.Button(form_frame, text="Save Annotation", command=self.save_annotation).grid(row=7, column=0, padx=10, pady=5)


        tk.Label(form_frame, text="Memo").grid(row=7, column=1, padx=10, pady=2)
        tk.Entry(form_frame, textvariable=self.text_content_var_1, width=40).grid(row=8, column=1, padx=10, pady=2)

        tk.Button(form_frame, text="Next Video", command=self.next_video).grid(row=8, column=0, padx=10, pady=5)

        # Log box at the bottom
        self.log = tk.Text(master, height=10, width=100)
        self.log.grid(row=10, column=0, columnspan=3, padx=20, pady=10)
        self.log.insert(tk.END, "Annotation log:\n")


        self.load_video()

    def load_video(self):
        if self.current_index >= len(self.video_files):
            self.video_label.config(text="No more videos.")
            return

        self.clear_fields()
        self.current_video = self.video_files[self.current_index]
        self.video_label.config(
            text=f"Video {self.current_index + 1} / {len(self.video_files)} — File: {self.current_video}")


    def clear_fields(self):
        self.violence_type_var.set("")
        self.video_type_var.set("")
        self.sound_violence_var.set("")
        self.sound_type_var.set("")
        self.text_violence_var.set("")
        self.text_content_var.set("")
        self.text_content_var_1.set("")
        self.video_start_var.set("")
        self.video_end_var.set("")

    def play_video(self):
        video_path = os.path.join(DOWNLOAD_DIR, self.current_video)
        os.startfile(video_path)  

    def save_annotation(self):
        start_time = self.video_start_var.get()
        end_time = self.video_end_var.get()

        if not start_time or not end_time:
            self.log.insert(tk.END, "Start and end time must be provided.\n")
            return

        number, title = self.current_video.split("_", 1)
        title = title.rsplit(".", 1)[0]

        row = {
            "No": int(number),
            "Title": title,
            "Filename": self.current_video,
            "Violence(Video) Type": self.violence_type_var.get(),
            "Video Start": start_time,
            "Video End": end_time,
            "Video Type": self.video_type_var.get(),
            "Violence(Sound) Type": self.sound_violence_var.get(),
            "Sound Start": start_time,
            "Sound End": end_time,
            "Sound Type": self.sound_type_var.get(),
            "Violence(Text) Type": self.text_violence_var.get(),
            "Text Start": start_time,
            "Text End": end_time,
            "Texts": self.text_content_var.get(),
            "Memos": self.text_content_var_1.get(),
        }

        file_exists = os.path.isfile(ANNOTATION_FILE)
        if file_exists:
            df = pd.read_excel(ANNOTATION_FILE)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_excel(ANNOTATION_FILE, index=False)
        self.log.insert(tk.END, "Annotation saved.\n")

    def next_video(self):
        self.current_index += 1
        self.log.delete(1.0, tk.END)
        self.log.insert(tk.END, "Annotation log:\n")
        self.load_video()


def list_downloaded_videos():
    if not os.path.exists(DOWNLOAD_DIR):
        return []
    return [f for f in sorted(os.listdir(DOWNLOAD_DIR)) if f.endswith(".mp4") and "_" in f]
