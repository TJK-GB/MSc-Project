import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import threading
import time

class VideoViewerByID:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Viewer by ID")
        self.root.geometry("1400x800")

        self.VIDEO_DIR = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\youtube_videos"
        self.BASE_PATH_SAVE = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\annotations\Original\Framelevel"
        self.SAVE_PATH = os.path.join(self.BASE_PATH_SAVE, "annotation_(framelevel)_ver2.xlsx")

        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False

        self.video_id_map = self.load_id_to_filename()
        self.video_ids = sorted(self.video_id_map.keys())
        self.current_video_index = 0
        self.current_filename = None

        self.setup_ui()

    def load_id_to_filename(self):
        df = pd.read_excel(self.SAVE_PATH)
        df = df.sort_values(by=["Video ID"])
        id_map = {}
        for vid in df["Video ID"].unique():
            match = df[df["Video ID"] == vid]["Filename"].values[0]
            id_map[vid] = match
        return id_map

    def setup_ui(self):
        tk.Label(self.root, text="Enter Video ID (1–300):").pack()
        self.id_entry = tk.Entry(self.root, width=10)
        self.id_entry.pack()

        button_container = tk.Frame(self.root)
        button_container.pack(pady=5)

        tk.Button(button_container, text="Load Video", command=self.load_video_by_id).pack(side=tk.LEFT, padx=10)
        tk.Button(button_container, text="Next Video ▶", command=self.next_video).pack(side=tk.LEFT, padx=10)

        self.frame_label = tk.Label(self.root, text="Frame: 0 / 0")
        self.frame_label.pack()

        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        controls = tk.Frame(self.root)
        controls.pack()

        tk.Button(controls, text="▶ Play", command=self.play_video).grid(row=0, column=0)
        tk.Button(controls, text="⏸ Pause", command=self.pause_video).grid(row=0, column=1)
        tk.Button(controls, text="⏪ 1f", command=lambda: self.step_frame(-1)).grid(row=0, column=2)
        tk.Button(controls, text="⏪ 5f", command=lambda: self.step_frame(-5)).grid(row=0, column=3)
        tk.Button(controls, text="⏪ 10f", command=lambda: self.step_frame(-10)).grid(row=0, column=4)
        tk.Button(controls, text="⏩ 1f", command=lambda: self.step_frame(1)).grid(row=0, column=5)
        tk.Button(controls, text="⏩ 5f", command=lambda: self.step_frame(5)).grid(row=0, column=6)
        tk.Button(controls, text="⏩ 10f", command=lambda: self.step_frame(10)).grid(row=0, column=7)

        jump_frame_container = tk.Frame(self.root)
        jump_frame_container.pack(pady=5)

        tk.Label(jump_frame_container, text="Go to Frame:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(jump_frame_container, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(jump_frame_container, text="Go", command=self.go_to_frame).pack(side=tk.LEFT)

    def load_video_by_id(self):
        vid_id_str = self.id_entry.get().strip()
        if not vid_id_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid number.")
            return
        vid_id = int(vid_id_str)
        if vid_id not in self.video_id_map:
            messagebox.showerror("Error", "Video ID not found in annotation file.")
            return

        self.current_video_index = self.video_ids.index(vid_id)
        self.load_video_by_filename(self.video_id_map[vid_id])

    def next_video(self):
        self.current_video_index += 1
        if self.current_video_index >= len(self.video_ids):
            messagebox.showinfo("Done", "No more videos.")
            return
        vid_id = self.video_ids[self.current_video_index]
        self.id_entry.delete(0, tk.END)
        self.id_entry.insert(0, str(vid_id))
        self.load_video_by_filename(self.video_id_map[vid_id])

    def load_video_by_filename(self, filename):
        path = os.path.join(self.VIDEO_DIR, filename)
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Video file not found: {filename}")
            return

        self.current_filename = filename
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.update_canvas()

    def update_canvas(self):
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if not ret:
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        scale = min(1000 / w, 600 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.imgtk = imgtk
        self.canvas.config(image=imgtk)

        time_secs = self.current_frame / self.fps
        mm, ss = divmod(time_secs, 60)
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}   |  Time: {int(mm):02}:{ss:.2f}")

    def play_video(self):
        if not self.cap or self.is_playing:
            return
        self.is_playing = True
        def loop():
            while self.is_playing:
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.total_frames - 1:
                    break
                self.update_canvas()
                time.sleep(1.0 / self.fps)
        threading.Thread(target=loop, daemon=True).start()

    def pause_video(self):
        self.is_playing = False

    def step_frame(self, frames):
        if not self.cap:
            return
        new_frame = max(0, min(self.current_frame + frames, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.update_canvas()
        self.is_playing = False

    def go_to_frame(self):
        if not self.cap:
            return
        try:
            frame_num = int(self.jump_entry.get().strip())
            if 0 <= frame_num < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                self.update_canvas()
                self.is_playing = False
            else:
                messagebox.showerror("Invalid Frame", f"Frame must be between 0 and {self.total_frames - 1}")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid frame number.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoViewerByID(root)
    root.mainloop()

