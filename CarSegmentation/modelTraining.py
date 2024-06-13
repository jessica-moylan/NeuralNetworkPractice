from tqdm import tqdm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from uNetModel import UNET
from utilities import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_images
)

#GLOBAL PARAMS
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "/home/jessica/Project Code/car_data/train_data/"
TRAIN_MASK_DIR = "/home/jessica/Project Code/car_data/train_masks_data/"
VALIDATION_IMG_DIR = "/home/jessica/Project Code/car_data/validation_images/"
VALIDATION_MASK_DIR = "/home/jessica/Project Code/car_data/validation_masks/"


def training_function(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_index, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        #moving forwards
        with torch.cuda.amp.autocast(): #ueses float16
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #mmoving backwards
        optimizer.zero_grad() #zeros the gradients from the previous
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss = loss.item())



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE) #if we wanted multisegmentation, we would change the number of output channels to the correct number of "classes"

    #binary cross entropy - we choose this beacuse we are not doing sigmoid for the output
    loss_fn = nn.BCEWithLogitsLoss() #we would change this to CEEithLogitsLoss() if multi classes
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VALIDATION_IMG_DIR,
        VALIDATION_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        training_function(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_images(val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()