
import onnx
import torch
import torch.backends.cudnn as cudnn

from argparse import ArgumentParser
from collections import OrderedDict
from lightning_model import LightningModel

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best.ckpt")
parser.add_argument("--batch", type=int, default=4)
args = parser.parse_args()

model = LightningModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
onnx_path = args.checkpoint_path.replace(".ckpt", ".onnx")

torch.onnx.export(model, dummy_input, onnx_path,
                export_params=True, verbose=False,
                input_names=["input"], 
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, 
                              "output": {0: "batch_size"}})

onnx_model = onnx.load(onnx_path)
onnx_model.metadata_props.append(onnx.StringStringEntryProto(key="names", value= str(model.class_names)))
onnx.save_model(onnx_model, onnx_path)
