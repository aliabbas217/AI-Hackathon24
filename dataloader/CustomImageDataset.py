import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import json
import yaml

class CustomImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        img_path = [path['file_name'] for path in self.dataset["images"] if path["id"] == idx][0]
        temp = img_path.split('/')[2] + '/' +'0000'+ img_path.split('/')[3][4:]
        img_path = 'data/gtea_png/png/'+temp
        image = read_image(img_path)
        # image = img_path
        annotations = [ann for ann in self.dataset["annotations"] if ann["image_id"] == idx]

        bbox = []
        keypoints = []
        mode = []
        category_id = []
        area = []
        iscrowd = []
        action = []
        num_keypoints = []
        with open('dataloader/dataset.yml', 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        for ann in annotations:
            bbox.append(ann['bbox'])
            keypoints.append(ann['keypoints'])
            mode.append(ann['mode'])
            category_id.append(ann['category_id'])
            area.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
            action.append(ann['action'])
            num_keypoints.append(ann['num_keypoints'])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            annotations = self.target_transform(annotations)
        return {"image": image, "bbox": bbox, "keypoints": keypoints, "mode": data["mode"][mode], "category_id": category_id, "area": area, "iscrowd": iscrowd, 
                "action": data["action_mapping"][action], "num_keypoints": num_keypoints}