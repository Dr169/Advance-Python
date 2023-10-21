import sys
import torch
import lightning.pytorch as pl
import torch.backends.cudnn as cudnn
from image_dataset import dataset
from lightning_model import LightningModel
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_path = sys.argv[1]
    val_path = sys.argv[2]

    train_dataset, val_dataset = dataset(train_data_path=train_path, val_data_path=val_path, is_not_test=True)

    logger = TensorBoardLogger("logs", name="CNN_model_train")
    model = LightningModel(num_classes=10)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints",
        filename="best",
        save_last=True
        )
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=20, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataset, val_dataset)