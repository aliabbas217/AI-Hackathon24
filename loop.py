import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.transforms import v2  # No longer needed if using albumentations
from arch.model import Net
from torchmetrics.detection import iou  # Use IntersectionOverUnion if using torchmetrics >= 0.11
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Accuracy, F1Score
from torch.utils.tensorboard import SummaryWriter
from utils.xml_loader import HandKeypointDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

with open('dataloader/dataset.yml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

batch_size = 8  
learning_rate = 1e-4  
num_epochs = 100  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Resize(data["albumentations"]["resize_height"], data["albumentations"]["resize_width"]), # Access from YAML
    A.Normalize(mean=data['mean'], std=data['std']),  # Access from YAML
    ToTensorV2()  # Convert to Tensor *after* other transforms
])

images_path = "data/gtea_png/png/"
xml_path = "data/xmls/"

train_dataset = HandKeypointDataset(image_dir=images_path+"train", xml_path=xml_path+"train", transform=transform,
                                    mode_map = {"left": 0, "right": 1},
                                    )
train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = HandKeypointDataset(image_dir=images_path+"test", xml_path=xml_path+"test", transform=transform,
                                   mode_map = {"left": 0, "right": 1},
                                   )
test_loader = DataLoader(test_dataset, batch_size=batch_size)


model = Net().to(device)

bbox_loss_fn = nn.MSELoss()
keypoint_loss_fn = nn.MSELoss()
mode_loss_fn = nn.BCELoss()
action_loss_fn = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter(log_dir="logs")

iou_metric = iou.IntersectionOverUnion()
map_metric = MeanAveragePrecision()
accuracy_metric_mode = Accuracy(task="binary")
f1_metric_mode = F1Score(task="binary")
accuracy_metric_action = Accuracy(task="multiclass", num_classes=10)
f1_metric_action = F1Score(task = 'multiclass', num_classes= 10)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    
    for i, (images, annotations) in enumerate(train_loader):  # Get images and annotations
        images = images.to(device)
        bboxes_list = torch.stack([torch.tensor(ann['boxes']['left'][0]['bbox'] + ann['boxes']['right'][0]['bbox'], dtype=torch.float32) for ann in annotations]).to(device)
        keypoints_list = [ann['keypoints'] for ann in annotations]
        modes = [ann['modes'] for ann in annotations]
        actions = [ann['actions'] for ann in annotations]

        bbox_pred, keypoint_pred, mode_pred, action_pred = model(images)

        bbox_loss = bbox_loss_fn(bbox_pred, bboxes_list)

        keypoint_loss = keypoint_loss_fn(keypoint_pred, keypoints_list)
        mode_loss = mode_loss_fn(mode_pred, modes)
        action_loss = action_loss_fn(action_pred, actions)

        loss = 0.5 * bbox_loss + 0.3 * keypoint_loss + 0.1 * mode_loss + 0.1 * action_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        iou_score = iou_metric(bbox_pred, torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in bboxes_list]).to(device))
        map_metric.update(bbox_pred, bboxes_list) 

        mode_accuracy = accuracy_metric_mode(mode_pred, modes.int())
        mode_f1 = f1_metric_mode(mode_pred, modes.int())

        action_accuracy = accuracy_metric_action(action_pred, actions)
        action_f1 = f1_metric_action(action_pred, actions)

        writer.add_scalar("Batch/Loss", loss.item(), epoch * total_batches + i)
        writer.add_scalar("Batch/IoU", iou_score.mean().item(), epoch * total_batches + i)
        writer.add_scalar("Batch/mAP", map_metric.compute().item(), epoch * total_batches + i)
        writer.add_scalar("Batch/Mode Accuracy", mode_accuracy.item(), epoch * total_batches + i)
        writer.add_scalar("Batch/Mode F1", mode_f1.item(), epoch * total_batches + i)
        writer.add_scalar("Batch/Action Accuracy", action_accuracy.item(), epoch * total_batches + i)
        writer.add_scalar("Batch/Action F1", action_f1.item(), epoch * total_batches + i)

        iou_metric.reset()
        accuracy_metric_mode.reset()
        f1_metric_mode.reset()
        accuracy_metric_action.reset()
        f1_metric_action.reset()

    avg_loss = running_loss / total_batches
    writer.add_scalar("Epoch/Loss", avg_loss, epoch + 1)

    epoch_map = map_metric.compute()
    writer.add_scalar("Epoch/mAP", epoch_map.item(), epoch + 1)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, mAP: {epoch_map:.4f}")

    map_metric.reset()


torch.save(model.state_dict(), 'gtea_model.pth')

writer.close()