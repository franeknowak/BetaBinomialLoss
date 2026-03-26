import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import random

class CachedDatasetTempMultilabel(torch.utils.data.Dataset):
    def __init__(self, cache_dir, label_criterion = (None, None)):
        label_idx, mode = label_criterion

        assert label_idx in (None, 0, 1, 2), \
            f"Invalid value for label criterion selection: {label_idx!r}. Expected 'None' if using standard label value or '0', '1', '2' if specifying the criterion."
        if label_idx is not None:
            assert mode in ('soft', 'hard'), \
                f"Invalid value for label criterion selection: {mode!r}. Expected 'soft' if using standard label value or 'hard' if majority rounding"
        
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        self.label_criterion = label_criterion

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        # Multilabel
        if self.label_criterion[0] is None:
            if self.label_criterion[1] == 'soft':
                return data['image'], data['label'], data['vid_id'], data['frame_id']
            elif self.label_criterion[1] == 'hard':
                return data['image'], torch.round(data['label']), data['vid_id'], data['frame_id']
        # Individual label
        else:
            if self.label_criterion[1] == 'soft':
                return data['image'], data['label'][self.label_criterion[0]], data['vid_id'], data['frame_id']
            elif self.label_criterion[1] == 'hard':
                return data['image'], torch.round(data['label'][self.label_criterion[0]]), data['vid_id'], data['frame_id']

class CachedDatasetEncoderPretraining(torch.utils.data.Dataset):
    def __init__(self, cache_dir, label_criterion = (None, None)):
        idx, mode = label_criterion

        assert idx in (None, 0, 1, 2), \
            f"Invalid value for label criterion selection: {idx!r}. Expected 'None' if using standard label value or '0', '1', '2' if specifying the criterion."
        assert mode in ('soft', 'hard'), \
            f"Invalid value for label criterion selection: {mode!r}. Expected 'soft' if using standard label value or 'hard' if majority rounding"
        
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        self.label_criterion = label_criterion

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        if self.label_criterion[0] is None:
            # Assume multilabel
            if self.label_criterion[1] == 'soft':
                return data['image'], data['label']
            elif self.label_criterion[1] == 'hard':
                return data['image'], torch.round(data['label'])
        else:
            # User specified criterion
            if self.label_criterion[1] == 'soft':
                return data['image'], data['label'][self.label_criterion[0]]
            elif self.label_criterion[1] == 'hard':
                return data['image'], torch.round(data['label'][self.label_criterion[0]])