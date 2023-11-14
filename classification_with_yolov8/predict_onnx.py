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
parser.add_argument("--predict_path", type=str, default="./runs/classify/predict/")
parser.add_argument("--imgsz", type=int, default=224)
args = parser.parse_args()

onnx_model = onnx.load(args.model_path)
ort_session = onnxruntime.InferenceSession(args.model_path)

for prop in onnx_model.metadata_props:
    if prop.key == "names":
        class_names = eval(prop.value)
        break

for img in glob.glob(f"{args.data}/*jpg"):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.imgsz, args.imgsz)) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    ort_outs = ort_session.run(None, {ort_session.get_inputs()[0].name: image})
    input_label = class_names[np.argmax(ort_outs[0])]

    pil_image = Image.open(img)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/tlwg/Loma.ttf", 15)
    draw.rectangle(draw.textbbox((0, 0), input_label, font=font) , fill="black")
    draw.text((0, 0), input_label, font=font, fill="white")
    pil_image.save( args.predict_path + img.split("/")[-1].replace(".jpg", "_predicted.jpg"))