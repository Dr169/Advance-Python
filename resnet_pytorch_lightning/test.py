import sys
import torch
import lightning.pytorch as pl
import torch.backends.cudnn as cudnn
from image_dataset import dataset
from lightning_model import LightningModel
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    test_path = sys.argv[1]

    test_dataset = dataset(test_data_path=test_path, is_not_test=False)

    logger = TensorBoardLogger("logs", name="CNN_model_test")
    model = LightningModel.load_from_checkpoint(
    checkpoint_path="./checkpoints/best.ckpt",)

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=20, logger=logger)
    trainer.test(model, test_dataset, ckpt_path="./checkpoints/best.ckpt")