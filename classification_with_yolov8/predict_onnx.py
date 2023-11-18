import os
import cv2
import glob
import onnx
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont


parser = ArgumentParser()
parser.add_argument("--data", type=str, default="../datasets/classify/predict")
parser.add_argument("--model_path", type=str, default="./runs/classify/train/weights/best.onnx")
parser.add_argument("--predict_path", type=str, default="./runs/classify/onnx_predict/")
parser.add_argument("--imgsz", type=int, default=224)
parser.add_argument("--batch", type=int, default=4)
args = parser.parse_args()

onnx_model = onnx.load(args.model_path)
ort_session = onnxruntime.InferenceSession(args.model_path)

for prop in onnx_model.metadata_props:
    if prop.key == "names":
        class_names = eval(prop.value)
        break

data_set = glob.glob(f"{args.data}/*.jpg")
batch_size = args.batch
batched_data = []
batch = []
for index, img in enumerate(data_set, 1):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.imgsz, args.imgsz)) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    batch.append(image)

    if index % batch_size == 0:
        batched_data.append(np.concatenate(batch, axis=0))
        batch = []

if batch:
    batched_data.append(np.concatenate(batch, axis=0))

ort_outs = []
for idx, batch in enumerate(batched_data, 1):
    ort_outs.append(ort_session.run(None, {ort_session.get_inputs()[0].name: batch}))

input_label = []
for batch in ort_outs:
    for img in batch[0]:
        input_label.append(class_names[np.argmax(img)])

if not os.path.exists(args.predict_path):
   os.makedirs(args.predict_path)

for idx, img in enumerate(data_set):
    pil_image = Image.open(img)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/tlwg/Loma.ttf", 15)
    draw.rectangle(draw.textbbox((0, 0), input_label[idx], font=font) , fill="black")
    draw.text((0, 0), input_label[idx], font=font, fill="white")
    pil_image.save( args.predict_path + img.split("/")[-1].replace(".jpg", "_predicted.jpg"))