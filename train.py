import os
import random
import time
from torchvision.transforms.functional import resize
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torch import nn
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision
from torch.optim import Adam
import cv2
import dadaptation


def format_float(number, decimal_places):
    # Format the float with the specified number of decimal places and zero padding
    formatted_string = "{:.{}f}".format(number, decimal_places)
    return formatted_string


class StatsTracker:
    def __init__(self, callback_list):
        self.dict_list = []
        for callback in callback_list:
            tmp = {}
            tmp['name'] = str(callback.__name__).replace("_metric", '')
            tmp['callback'] = callback
            tmp['count'] = 0
            tmp['rolling_average'] = 0
            tmp['imm'] = 0
            tmp['status'] = 'start'
            self.dict_list += [tmp]
        self.dict_dict = {}
        for i, dict in enumerate(self.dict_list):
            key = dict['name']
            self.dict_dict[key] = dict

    def add_info(self, output, target):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(output, target.to(torch.long), mode='binary', threshold=0.5)

        for key in self.dict_dict:
            value = self.dict_dict[key]['callback'](tp, fp, fn, tn, reduction="micro")
            self.dict_dict[key]['imm'] = value
            if self.dict_dict[key]['status'] == 'start':
                self.dict_dict[key]['status'] = 'running'
                self.dict_dict[key]['rolling_average'] = value
                self.dict_dict[key]['count'] = 1
            else:
                current_average = self.dict_dict[key]['rolling_average']
                new_average = (current_average * self.dict_dict[key]['count'] + value) / (
                        self.dict_dict[key]['count'] + 1)
                self.dict_dict[key]['rolling_average'] = new_average
                self.dict_dict[key]['count'] = self.dict_dict[key]['count'] + 1

    def get_info(self):
        mess = 'Stats Averages[ '
        for key in self.dict_dict:
            mess += ' ' + self.dict_dict[key]['name'] + ':' + format_float(self.dict_dict[key]['rolling_average'], 4)
        mess += ' ] '
        mess += ' Immediate[ '
        for key in self.dict_dict:
            mess += ' ' + self.dict_dict[key]['name'] + ':' + format_float(self.dict_dict[key]['imm'], 4)
        mess += ' ] '
        return mess

    def clear(self):
        for key in self.dict_dict:
            tmp = {'count': 0, 'rolling_average': 0, 'status': 'start', 'imm': 0}
            self.dict_dict[key].update(tmp)

    def get_specific(self, key):
        return self.dict_dict[key]['rolling_average']


def find_square_bounds(prearray):
    array = prearray[:, :, 0]
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    # y1, y2 = np.where(rows)[0][[0, -1]]
    # x1, x2 = np.where(cols)[0][[0, -1]]
    x1 = np.argmax(cols)
    y1 = np.argmax(rows)

    x2 = len(cols) - np.argmax(cols[::-1])
    y2 = len(rows) - np.argmax(rows[::-1])

    return x1, y1, x2, y2


class ToCuda:
    def __call__(self, tensor):
        return tensor.to('cuda') if torch.cuda.is_available() else tensor


class RandomScaleAndResize:
    def __init__(self, scale=(0.3, 0.9), probability=0.5):
        self.scale = scale
        self.probability = probability

    def __call__(self, img):
        # 50% chance to apply the transformation
        if random.random() < self.probability:
            return img
        # get the original size (height, width) of the image
        _, height, width = img.size()
        original_size = (height, width)

        # calculate scaled dimensions
        scale = random.uniform(*self.scale)
        scaled_size = (int(height * scale), int(width * scale))

        # downscale and then upscale back to original size
        img = resize(img, scaled_size)
        img = resize(img, original_size)

        return img


# Define custom dataset
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_list, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = img_list
        self.transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.transform_augment = Compose(
            [ToTensor(), ToCuda(), RandomScaleAndResize(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.rotate = False

        self.augment = augment
        self.look_up_dict = {}
        print("running pre image loading")
        pbar = tqdm(total=len(self.img_list))
        for idx, img in enumerate(self.img_list):
            img_name = self.img_list[idx]
            img_path = os.path.join(self.img_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name)
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            mask_bool = mask == 255
            x1, y1, x2, y2 = find_square_bounds(mask_bool)
            height, width, _ = img.shape
            tmp = {}
            tmp['shape'] = (height, width, _)
            tmp['bounds'] = (x1, y1, x2, y2)
            #img=cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
            tmp['img'] = img
            tmp['mask'] = mask[:, :, 0]
            self.look_up_dict[idx] = tmp
            pbar.update(1)

            # max_left = x1
            # max_right = width - x2
            # max_up = y1
            # max_down = height - y2
            # roll_x = np.random.randint(-max_left, max_right)
            # roll_y = np.random.randint(-max_up, max_down)
            # img = np.roll(img, roll_y, axis=0)
            # img = np.roll(img, roll_x, axis=1)
            # mask = np.roll(mask, roll_y, axis=0)
            # mask = np.roll(mask, roll_x, axis=1)

        pbar.close()

    def __len__(self):
        return len(self.img_list)

    def toggle(self):
        self.rotate = not self.rotate

    def __getitem__(self, idx):
        tmp = self.look_up_dict[idx]
        img = np.copy(tmp['img'])
        mask = np.copy(tmp['mask'])

        if np.random.rand() < 0.5:

        elif self.augment:
            height, width, _ = tmp['shape']
            x1, y1, x2, y2 = tmp['bounds']
            max_left = x1
            max_right = width - x2
            max_up = y1
            max_down = height - y2
            roll_x = np.random.randint(-max_left, max_right)
            roll_y = np.random.randint(-max_up, max_down)
            img = np.roll(img, roll_y, axis=0)
            img = np.roll(img, roll_x, axis=1)
            mask = np.roll(mask, roll_y, axis=0)
            mask = np.roll(mask, roll_x, axis=1)

            if np.random.rand() < 0.5:
                if roll_y > 0:
                    img[0:roll_y] = 255
                elif roll_y < 0:
                    img[height + roll_y:height] = 255

                if roll_x > 0:
                    img[:, 0:roll_x] = 255
                elif roll_x < 0:
                    img[:, width + roll_x:width] = 255

            if np.random.rand() < 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            #if self.rotate:
            #    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            #    mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        if self.augment:
            img = Image.fromarray(img)
            img = self.transform_augment(img)
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            mask = mask.cuda()
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            img = self.transform(img)
            img = img.cuda()
            mask = mask.cuda()
        return img, mask


# Constants
IMG_DIR = "E:\\IMAGES"
MASK_DIR = "E:\\MASKS"
IMG_HEIGHT = 256
IMG_WIDTH = 512
BATCH_SIZE = 32
EPOCHS = 120

# Load image list
img_list = os.listdir(IMG_DIR)
np.random.seed(42)
np.random.shuffle(img_list)
train_list, val_list = train_test_split(img_list, test_size=0.2, random_state=42)

# Prepare data loaders
train_dataset = SegmentationDataset(IMG_DIR, MASK_DIR, train_list, augment=True)
val_dataset = SegmentationDataset(IMG_DIR, MASK_DIR, val_list)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
# model = UnetPlusPlus('vit_base_patch16_224', classes=1, activation='sigmoid')

encoder_name = "mit_b2"
encoder_weights = "imagenet"
classes = 1

model = smp.Unet(
    in_channels=3,
    encoder_name=encoder_name,
    encoder_weights=encoder_weights,
    classes=classes,
    activation='sigmoid'
)

model = model.cuda()

# Define loss
dice_loss = smp.losses.DiceLoss(mode='binary')
# iou_loss = loss = smp.losses.(mode='binary')
# Define loss
loss = dice_loss

# Initialize optimizer
accumulation_steps = 2
stat_tracker = StatsTracker([smp.metrics.iou_score, smp.metrics.f1_score, smp.metrics.accuracy])
max_iou = stat_tracker.get_specific('iou_score')
# optimizer = Adam(model.parameters(), lr=0.001)
optimizer = dadaptation.DAdaptAdam(model.parameters(), lr=1, weight_decay=.01)
# Train the model

for epoch in range(EPOCHS):
    print(f"training epoch {epoch}")
    model.train()
    pbar = tqdm(total=len(train_loader))
    last_index = 0
    optimizer.zero_grad()
    for i, (imgs, masks) in enumerate(train_loader):
        last_index = i
        preds = model(imgs)
        loss_value = loss(preds, masks)

        # optimizer.zero_grad()
        # loss_value.backward()
        # optimizer.step()
        loss_value.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


        stat_tracker.add_info(preds, masks)
        message = stat_tracker.get_info()
        pbar.update(1)
        pbar.set_postfix({'Additional Info': message}, refresh=True)

    # After the loop, check if there's an unexecuted gradient accumulation step
    if (last_index + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    pbar.close()
    time.sleep(.1)

    model.eval()
    print("running validation ")
    stat_tracker.clear()
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader):
            preds = model(imgs)
            val_loss_value = loss(preds, masks)
            stat_tracker.add_info(preds, masks)
    time.sleep(.1)
    iou = stat_tracker.get_specific('iou_score')
    f1 = stat_tracker.get_specific('f1_score')
    if iou > max_iou:
        max_iou = iou
        save_model = f'BarSegViT_epoch_{epoch}_{iou}_.pt'
        torch.save(model, save_model)
        print(f'saving model at {save_model}')

    print(f"Epoch: {epoch}, F1: {f1}, IOU: {iou}")
    stat_tracker.clear()
