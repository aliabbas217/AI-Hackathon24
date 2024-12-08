import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToPILImage
from CustomImageDataset import CustomImageDataset


dataset_path = "data/merged_final.json"


dataset = CustomImageDataset(dataset_path=dataset_path)

random_idx = random.randint(0, len(dataset) - 1)

sample = dataset[random_idx]

image = ToPILImage()(sample["image"])

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.axis('off')
ax.set_title(f"Image Index: {random_idx}")

for bbox in sample['bbox']:

    x_min, y_min, width, height = bbox
    rect = patches.Rectangle(
        (x_min, y_min), width, height, 
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)


for keypoints in sample['keypoints']:
    for i in range(0, len(keypoints), 3):
        x, y, visibility = keypoints[i:i+3]
        if visibility > 0:
            ax.plot(x, y, 'bo', markersize=3)

plt.show()