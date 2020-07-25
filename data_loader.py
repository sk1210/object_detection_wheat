from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import cv2
import csv
import os
from transforms import ToTensor
from torchvision.transforms import functional as F
import transforms


class WheatData(Dataset):
    def __init__(self, csv_path=""):
        self.csv_path = r"kaggle\train.csv"
        self.images_path = r"kaggle\input\global-wheat-detection\train"

        self.transforms = transforms.RandomHorizontalFlip(0.5)
        self.images_info = {}
        self.total_samples = None
        self.scaleFactor = 0.5

        self.read_csv()
        self.key_list = list(self.images_info.keys())

    def write_images(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for key, data in self.images_info.items():
            img_path = os.path.join(self.images_path, key + ".jpg")
            boxes = data["bbox"]

            img = cv2.imread(img_path)

            # draw boxes
            for box in boxes:
                x1, y1, w, h = list(map(int, box))
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 100, 0), 3)
                cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), img)

    def read_csv(self):
        print("Loading CSV ")
        with open(self.csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                image_id = row[0]
                if not (image_id in self.images_info):
                    self.images_info[image_id] = {}

                for key, value in zip(header, row):
                    if key == "bbox":
                        box = value.replace("[", "").replace("]", "").replace(" ", "").split(",")
                        x1, y1, w, h = list(map(float, box))
                        box = [x1, y1, x1 + w, y1 + h]
                        if key in self.images_info[image_id]:
                            self.images_info[image_id][key].append(box)
                        else:
                            self.images_info[image_id][key] = [box]
                    else:
                        self.images_info[image_id][key] = value
                # print (self.images_info)
        self.total_samples = len(self.images_info)

    def apply_transorms(self, img, target):

        transorm_flip = transforms.RandomHorizontalFlip(0.5)
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2, hue=0.1)

        img = jitter(img)
        img = F.to_tensor(img)
        img, target = transorm_flip(img, target)
        #img = normalize(img)

        return img, target

    def __getitem__(self, idx):
        key = self.key_list[idx]
        data = self.images_info[key]
        img_path = os.path.join(self.images_path, key + ".jpg")
        boxes = data["bbox"]
        # img = cv2.imread(img_path)/255.0
        img = Image.open(img_path).convert("RGB")
        width, height = int(img.width * self.scaleFactor), int(img.height * self.scaleFactor)
        img = img.resize((width, height))
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32) * self.scaleFactor
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = key
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.apply_transorms(img, target)

        return img, target

    def __len__(self):
        return self.total_samples


if __name__ == "__main__":
    dataset = WheatData()
    print("END")
    # image_data.write_images("kaggle/temp/output")

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(next(iter(dataloader)))
