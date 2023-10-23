import glob
import torch
import lightning.pytorch as pl
import torch.backends.cudnn as cudnn

from image_dataset import dataset
from argparse import ArgumentParser
from lightning_model import LightningModel
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default="./Dataset/Train")
    parser.add_argument("--val_data", type=str, default="./Dataset/Validation")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)
    args = parser.parse_args()

    class_names = glob.glob(f"{args.train_data}/*")
    class_names = list(map(lambda x: x.split("/")[-1], class_names))

    train_dataset, val_dataset = dataset(
        train_data_path=args.train_data, 
        val_data_path=args.val_data, 
        is_test=False, 
        class_names=class_names)

    logger = TensorBoardLogger("logs", name="CNN_model_train")
    model = LightningModel(num_classes=args.num_classes, class_names=class_names)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints",
        filename="best",
        save_last=True
        )
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda:0" else "cpu", 
        devices=args.devices, 
        max_epochs=args.max_epochs, 
        logger=logger, 
        callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataset, val_dataset)