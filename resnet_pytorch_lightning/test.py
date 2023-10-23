import torch
import lightning.pytorch as pl
import torch.backends.cudnn as cudnn

from image_dataset import dataset
from argparse import ArgumentParser
from lightning_model import LightningModel
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = ArgumentParser()
    parser.add_argument("--test_data", type=str, default="./Dataset/Test")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best.ckpt")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)
    args = parser.parse_args()

    logger = TensorBoardLogger("logs", name="CNN_model_test")
    model = LightningModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    model.eval()
    model.freeze()
    test_dataset = dataset(test_data_path=args.test_data, is_test=True, class_names=model.class_names)

    trainer = pl.Trainer(accelerator="gpu" if device == "cuda:0" else "cpu", devices=args.devices, max_epochs=args.max_epochs, logger=logger)
    trainer.test(model, test_dataset, ckpt_path=args.checkpoint_path)