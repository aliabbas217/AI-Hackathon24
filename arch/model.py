import torch
import torch.nn as nn
import torch.nn.functional as F
from arch.backbone import DLANet

class Net(nn.Module):
    def __init__(self, num_classes = 10, num_keypoints = 21):
        super().__init__()
        self.num_classes = num_classes
        self.dla60 = DLANet(dla='dla60', return_index=[1, 2, 3, 4, 5])

        self.neck = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(16 * 16, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU()
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.Linear(4096, 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256, 4)
        )

        self.keypoint_regressor = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Linear(2048, 4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU(),
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, num_keypoints*3)
        )

        self.transformer = nn.Transformer(
            d_model=num_keypoints*3,
            nhead=3,
            dim_feedforward=1024,
            dropout=0.1
        )

        self.mode_classifier = nn.Sequential(
            nn.Linear(num_keypoints*3, 256),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.previous_x2 = None

    def forward(self, x):
        self.dla60.eval()
        x = self.dla60(x)
        x = self.neck(x)

        x1 = self.bbox_regressor(x)
        x2 = self.keypoint_regressor(x)

        x3 = self.mode_classifier(x2)

        batch_size = x2.shape[0]
        seq_len = x2.size(1)

        x2_for_transformer = x2.view(batch_size, seq_len, -1).permute(1, 0, 2)

        if self.previous_x2 is None:
            x4 = self.transformer(x2_for_transformer, x2_for_transformer)
        else:
            prev_x2 = self.previous_x2.permute(1, 0, 2)
            x4 = self.transformer(x2_for_transformer, prev_x2)

        x4 = x4.permute(1, 0, 2).squeeze(1)

        x4 = nn.Linear(x4.shape[0], self.num_classes)

        self.previous_x2 = x2.detach().clone()

        return x1, x2, x3, x4