import os
import shutil
from glob import glob

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import cv2
import json
import numpy as np


def read_image(img, label):
    image = cv2.imread(img)
    mask = np.zeros(image.shape, dtype=np.uint8)
    quad = json.load(open(label, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)
    cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, (256, 256))
    image = cv2.resize(image, (256, 256))
    return image, mask


def download_save_numpy(download_links, zip_data_path, numpy_data_path):
    def get_file_name(full_path):
        return full_path.split("/")[-1]

    for i, link in enumerate(download_links):
        file_name = get_file_name(link)
        os.system(f"wget {link} -P {zip_data_path}")
        os.system(f"unzip {zip_data_path}/{file_name} -d {zip_data_path}")
        imgdir_path = f"{zip_data_path}/{file_name.replace('.zip', '')}/images/"
        label_path = f"{zip_data_path}/{file_name.replace('.zip', '')}/ground_truth/"
        X = []
        Y = []
        for images, ground_truth in tqdm(zip(sorted(os.listdir(imgdir_path)), sorted(os.listdir(label_path)))):
            img_list = sorted(glob(imgdir_path + images + '/*.tif'))
            label_list = sorted(glob(label_path + ground_truth + '/*.json'))
            for img, label in zip(img_list, label_list):
                image, mask = read_image(img, label)
                X.append(image)
                Y.append(mask)

        X = np.array(X)
        Y = np.array(Y)
        Y = np.expand_dims(Y, axis=3)
        # print(X.shape, Y.shape)
        np.save(f"{numpy_data_path}/train_image{str(i)}.npy", X)
        np.save(f"{numpy_data_path}/mask_image{str(i)}.npy", Y)
        print('Files Saved For:', link[40:].replace('.zip', ''))
        os.remove(f"{zip_data_path}/{file_name}")
        shutil.rmtree(f"{zip_data_path}/{file_name.replace('.zip', '')}")


def combine_and_write_numpy(input_dir, output_dir):
    files = os.listdir(input_dir)
    image_files = sorted(filter(lambda x: x.startswith("train_image"), files))
    mask_files = sorted(filter(lambda x: x.startswith("mask_image"), files))
    for i, file in tqdm(enumerate(image_files)):
        if i == 0:
            total = np.load(f"{input_dir}/{file}")
        else:
            temp = np.load(f"{input_dir}/{file}")
            total = np.vstack((total, temp))
    np.save(f'{output_dir}/final_image.npy', total)

    for i, file in tqdm(enumerate(mask_files)):
        if i == 0:
            total = np.load(f"{input_dir}/{file}")
        else:
            temp = np.load(f"{input_dir}/{file}")
            total = np.vstack((total, temp))
    np.save(f'{output_dir}/final_mask.npy', total)


def load_train_val_test_data_with_split(data_path, val_fraction=0.11, test_fraction=0.1, batch_size=16):

    X = np.load(f"{data_path}/final_image.npy")
    Y = np.load(f"{data_path}/final_mask.npy")
    Y = Y.astype(bool).astype(int)

    ## Using fraction only for local machine
    ## Delete these  lines when training on the server
    idx = np.random.randint(0, X.shape[0], 2000)
    X = X[idx]
    Y = Y[idx]
    ## End of lines to be deleted
    print(X.shape, Y.shape)
    pass
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        shuffle=True,
                                                        random_state=42,
                                                        test_size=test_fraction)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      shuffle=True,
                                                      random_state=42,
                                                      test_size=val_fraction)
    X_train_tensor = torch.Tensor(X_train)
    X_val_tensor = torch.Tensor(X_val)
    X_test_tensor = torch.Tensor(X_test)

    y_train_tensor = torch.Tensor(Y_train)
    y_val_tensor = torch.Tensor(Y_val)
    y_test_tensor = torch.Tensor(Y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def calculate_iou(gt_mask, pred_mask, threshold=0.5):
    gt_mask = gt_mask.detach().numpy()
    pred_mask = pred_mask.detach().numpy()
    pred_mask = pred_mask.round()
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou