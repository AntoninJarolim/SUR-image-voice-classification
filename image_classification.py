import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets
from torchvision.transforms import v2
import torchvision.transforms.functional
from torchvision.io import read_image
import torchvision.io
from pathlib import Path
import argparse
import sys

to_torch = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

class Eye(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        #40x40
        self.conv2a = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2b = nn.Conv2d(16, 32, 3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(32)
        #20x20
        self.conv3a = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3b = nn.Conv2d(32, 64, 3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(64)
        #10x10

        self.conv4a = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4b = nn.Conv2d(64, 64, 3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(64)
        # 5x5

        self.lin1 = nn.Linear(5*5*64, 64)
        self.lin2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        #40x40
        x = self.conv2a(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.pool1(x)
        #20x20
        x = self.conv3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.pool1(x)
        #10x10
        x = self.conv4a(x)
        x = F.relu(x)
        x = self.conv4b(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        x = self.pool1(x)
        #5x5x128

        x = torch.flatten(x, 1)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

def classify_images(data_path:Path, model_path:Path = Path("image_model.pth")):
    if not model_path.exists():
        print("Model not found", file=sys.stderr)
        return
    if not data_path.exists():
        print("Data not found", file=sys.stderr)
        return
    if len([f for f in data_path.iterdir() if f.is_file()]) == 0:
        print("No images found in data directory", file=sys.stderr)
        return

    model = Eye()
    model.load_state_dict(torch.load(str(model_path)))
    with torch.no_grad():
        for image_path in data_path.iterdir():
            if image_path.suffix != ".png":
                continue

            image = read_image(str(image_path), torchvision.io.ImageReadMode.RGB)
            image = to_torch(image).unsqueeze(0)
            ev = model(image)[0][0]
            print(image_path.stem, "%f" % ev, 1 if ev >= 0.5 else 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-data-path", type=Path)
    parser.add_argument("--img-model-path", type=Path, default=Path("models/image_model.pth"))
    args = parser.parse_args()
    classify_images(args.data_path, args.model_path)
