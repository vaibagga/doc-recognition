import itertools
import sys

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from models import UNet, MaskRCNN
from config import *
from model_config import SEG_MODELS, seg_model_configs, NUM_EPOCHS, seg_model_configs_test
from utils import load_train_val_test_data_with_split, calculate_iou


def main():
    train, val, test = load_train_val_test_data_with_split(NUMPY_COMBINE_PATH)
    for model_name in SEG_MODELS:
        model = None
        if model_name == "UNET":
            model = UNet()
        elif model_name == "MASKRCNN":
            model = MaskRCNN()
        optimizer = None
        criterion = BCEWithLogitsLoss()
        for optim, lr, batch_size in itertools.product(seg_model_configs_test["optimizers"],
                                                       seg_model_configs_test["learning_rate"],
                                                       seg_model_configs_test["batch_size"]):
            if optim == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            elif optim == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            for epoch in range(NUM_EPOCHS):
                model.train()
                running_loss = 0.0
                epoch_loss = None
                for inputs, labels in tqdm(train, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
                    optimizer.zero_grad()
                    inputs = inputs.permute(0, 3, 1, 2)
                    labels = labels.permute(0, 3, 1, 2)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    epoch_loss = running_loss / len(train.dataset)
                    #print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
                running_iou = 0.0
                running_loss = 0.0
                for inputs, labels in tqdm(val, desc="Running validation predictions"):
                    #optimizer.zero_grad()
                    inputs = inputs.permute(0, 3, 1, 2)
                    labels = labels.permute(0, 3, 1, 2)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    iou_batch = calculate_iou(outputs, labels)
                    running_iou += iou_batch * inputs.size(0)
                print(f"Average val loss = {running_loss/len(val.dataset)}")
                print(f"Runnning IoU = {running_iou / len(val.dataset)}")



            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/{model_name}_{optim}_{batch_size}_{lr}.pth")


if __name__ == "__main__":
    main()
