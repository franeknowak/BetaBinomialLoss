import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import json
class EndoscapesDataset(Dataset):
    def __init__(self,
                 split,
                 transforms,
                 temporal,
                 label_criterion = (None, None),
                 annotations_path = 'config/reformatted_annotations_frames.json',
                 dataset_dir = '../dataset/Endoscapes'):
        
        label_idx, mode = label_criterion
        assert label_idx in (None, 0, 1, 2), \
            f"Invalid value for label criterion selection: {label_idx!r}. Expected 'None' if using standard label value or '0', '1', '2' if specifying the criterion."
        assert mode in ('soft', 'hard'), \
            f"Invalid value for label criterion selection: {mode!r}. Expected 'soft' if using standard label value or 'hard' if majority rounding"
        
        with open(annotations_path) as f:
            annotations = json.load(f)[split] # Split disclosed here
        
        self.split = split
        self.label_criterion = label_criterion
        self.temporal = temporal
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.keys = list(annotations.keys())
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)
    
    def load_frame(self, filename):
        path_to_image = Path(self.dataset_dir) / self.split / filename
        img = Image.open(path_to_image).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img

    def __getitem__(self, idx):
        data_point = self.annotations[self.keys[idx]]

        frames = data_point['frames']
        ann = data_point['annotations']
        label = torch.tensor(ann['ds'], dtype=torch.float32)
        if self.label_criterion[1] == 'hard':
            label = torch.round(label)
        if self.label_criterion[0] is not None:
            label = label[self.label_criterion[0]]

        vid_id = ann['video_id']
        frame_id = ann['frame_id']

        if self.temporal:
            image = torch.stack([self.load_frame(f) for f in frames], dim=0)  # [T, C, H, W]
        else:
            image = self.load_frame(frames[-1])  # [C, H, W]

        return image, label, vid_id, frame_id