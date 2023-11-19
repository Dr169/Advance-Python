import os
import cv2
import glob
import onnx
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont


parser = ArgumentParser()
parser.add_argument("--data", type=str, default="../datasets/detection/predict")
parser.add_argument("--model_path", type=str, default="./runs/detect/train/weights/best.onnx")
parser.add_argument("--predict_path", type=str, default="./runs/detect/onnx_predict")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=4)
args = parser.parse_args()


onnx_model = onnx.load(args.model_path)
ort_session = onnxruntime.InferenceSession(args.model_path)

for prop in onnx_model.metadata_props:
    if prop.key == "names":
        class_names = eval(prop.value)
        break

data_set = glob.glob(f"{args.data}/*.jpg")
image_size = args.imgsz
batch_size = args.batch
batched_data = []
batch = []

for index, img in enumerate(data_set, 1):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size)) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    batch.append(image)

    if index % batch_size == 0:
        batched_data.append(np.concatenate(batch, axis=0))
        batch = []

if batch:
    batched_data.append(np.concatenate(batch, axis=0))

ort_outs = []
for batch in batched_data:
    ort_outs.append(ort_session.run(None, {ort_session.get_inputs()[0].name: batch})[0])

for batch_out in ort_outs:
    for i in range(batch_out.shape[0]):
        pred = batch_out[i]
        img_path = data_set[i]

        original_image = Image.open(img_path)
        draw = ImageDraw.Draw(original_image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/tlwg/Loma.ttf", 20)


        original_width, original_height = original_image.size
        target_width, target_height = image_size, image_size
        width_coefficient = original_width / target_width
        height_coefficient = original_height / target_height


        anchor_scores = np.max(pred[4:, :], axis=0)
        keep = anchor_scores > 0.25
        pred = pred[: , keep]

        num_classes, num_anchors = pred.shape[0] - 4, pred.shape[1]

        bbox_coords = np.transpose(pred[:4])
        class_probs = np.transpose(pred[4:])

        keep_indices = cv2.dnn.NMSBoxes(bbox_coords, anchor_scores[keep], 0.25, 0.4)

        bboxes, class_labels, scores = [], [], []

        for idx in keep_indices:
            class_idx = np.argmax(class_probs[idx])

            score = class_probs[idx, class_idx]

            x, y, w, h = bbox_coords[idx]
            x1, y1 = float((x - w / 2)*width_coefficient), float((y - h / 2)*height_coefficient)
            x2, y2 = float((x + w / 2)*width_coefficient), float((y + h / 2)*height_coefficient)

            bboxes.append((x1, y1, x2, y2))
            class_labels.append(class_names[class_idx])
            scores.append(score)

        if not os.path.exists(args.predict_path):
            os.makedirs(args.predict_path)

        for bbox, class_label, score in zip(bboxes, class_labels, scores):
            draw.rectangle(bbox, outline="red", width=5)
            draw.rectangle(draw.textbbox((bbox[0]+5, bbox[1]-5), f"{class_label}: {score:.2f}", font=font) , fill="red")
            draw.text((bbox[0]+5, bbox[1]-5), f"{class_label}: {score:.2f}", fill="white", font=font)

        original_image.save(f"{args.predict_path}/predicted_{i}.jpg")