import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ViolenceDataset(Dataset):
    def __init__(self, csv_path, npy_dir, balance=False):
        self.npy_dir = npy_dir
        self.data = pd.read_csv(csv_path)
        self.data = self.data[
        self.data['Violence Type (Video)'].str.lower().isin(['violent', 'non-violent'])
        ].reset_index(drop=True)

        if balance:
            
            violent_data = self.data[self.data['Violence Type (Video)'].str.lower() == 'violent']
            non_violent_data = self.data[self.data['Violence Type (Video)'].str.lower() == 'non-violent']
            non_violent_sampled = non_violent_data.sample(n=len(violent_data), random_state=42)
            self.data = pd.concat([violent_data, non_violent_sampled]).sample(frac=1, random_state=42)

        self.data = self.data.reset_index(drop=True)

 
        self.filenames = self.data['Segment ID'].astype(str) + '.npy'
        self.labels = self.data['Violence Type (Video)'].str.lower().map({
            'violent': 1,
            'non-violent': 0
        }).astype(int).values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.npy_dir, self.filenames[idx])
        video = np.load(npy_path).astype(np.float32) / 255.0  # Shape: [320, 224, 224, 3]
        video = torch.from_numpy(video).permute(0, 3, 1, 2)   
        if __debug__:
            print(f"[DEBUG] Index: {idx}, Label: {self.labels[idx]}, Type: {type(self.labels[idx])}")

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label











# 1st try
# Create dataset for model training

# import os
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# import torch

# # Class definition
# class ViolenceDataset(Dataset):
#     # load csv_files and mapping of each segment in numbers
    
#     def __init__(self, csv_file, frame_root_dir, transform = None, label_map = None):
#         self.data = pd.read_csv(csv_file)
#         self.frame_root_dir = frame_root_dir
#         self.transform = transform
#         # let's ignore not applicable, only violent and non-violent
#         self.data = self.data[self.data['Violence Type (Video)'].str.lower().isin(['violent', 'non-violent'])]
        
#         # Since non-violent(1,424 segments), violent(591 segments), n/a(309 segments), balancing the dataset is required
#         violent_data = self.data[self.data['Violence Type (Video)'].str.lower() == 'violent']
#         non_violent_data = self.data[self.data['Violence Type (Video)'].str.lower() == 'non-violent']
#         non_violent_sampled = non_violent_data.sample(n=len(violent_data), random_state=42)
#         self.data = pd.concat([violent_data, non_violent_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        
#         if label_map is None:
#             self.label_map = {'violent': 1, 'non-violent': 0, 'none': -1}

#         else:
#             self.label_map = label_map

#     # finds the number of segments.
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         segment_id = row['Segment ID']
#         label_str = str(row['Violence Type (Video)']).strip().lower()

#         label = self.label_map.get(label_str, -1)   # -> unknown labels are -1
#         segment_path = os.path.join(self.frame_root_dir, segment_id)
#         max_frames = 64
#         frames = sorted([
#             os.path.join(segment_path, f)
#             for f in os.listdir(segment_path)
#             if f.endswith('.jpg')
#         ])[:max_frames]

#         images = []
#         for f in frames:
#             img = Image.open(f).convert('RGB')
#             if self.transform:
#                 img = self.transform(img)
#             else:
#                 img = transforms.ToTensor()(img)
#             images.append(img)

#         # For tensor [T, C, H, W]
#         video_tensor = torch.stack(images)

#         return video_tensor, label



# """ Configuration part"""
# # 1. resize and convert to tensor
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # 2. Directory management
# CSV_BASE_FILE = r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\Data split"
# train_csv = os.path.join(CSV_BASE_FILE, 'train.csv')
# val_csv = os.path.join(CSV_BASE_FILE, 'val.csv')
# test_csv = os.path.join(CSV_BASE_FILE, 'test.csv')
# frame_root_dir=r"C:\Users\a9188\Documents\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\Frames"


# # 3. Datasets to load
# train_dataset = ViolenceDataset(train_csv, frame_root_dir, transform=transform)
# val_dataset = ViolenceDataset(val_csv, frame_root_dir, transform=transform)
# test_dataset = ViolenceDataset(test_csv, frame_root_dir, transform=transform)


# # 4. Sample testing
# print(f'Loaded, {len(train_dataset)}, training samples')
# frames, label = train_dataset[0]
# print(f'Shape:, {frames.shape}')
# print(f'Label:, {label}')
