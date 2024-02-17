import cv2
import glob
import onnx
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont

onnx_model = onnx.load("./runs/segment/train/weights/best.onnx")
ort_session = onnxruntime.InferenceSession("./runs/segment/train/weights/best.onnx")
class_names = eval([prop.value for prop in onnx_model.metadata_props if prop.key == "names"][0])
data_set = glob.glob(f"../datasets/segmentation/test/images/*.jpg")[:10]
image_size = 640
batch_size = 4

def predict(batched_data, idx):
    ort_outs = ort_session.run(None, {ort_session.get_inputs()[0].name: np.concatenate(batched_data, axis=0)})
    for i in range(ort_outs[0].shape[0]):
        output0 = ort_outs[0][i].transpose()
        output1 = ort_outs[1][i]
        masks = output0[:,4+len(class_names):]
        boxes = output0[:, :4+len(class_names)]
        masks = masks @ output1.reshape(32, 160 * 160)
        boxes = np.hstack((boxes, masks))
        original_image = Image.open(data_set[i + idx - (batch_size)])
        draw = ImageDraw.Draw(original_image, "RGBA")
        font = ImageFont.truetype("/usr/share/fonts/truetype/tlwg/Loma.ttf", 20)
        original_width, original_height = original_image.size

        bbox_coords, anchor_scores, prediction_values = [], [], []
        for row in boxes:
            prob = row[4:4 + len(class_names)].max()
            if prob < 0.25:
                continue

            xc, yc, w, h = row[:4]
            label = class_names[row[4:4 + len(class_names)].argmax()]
            x1, y1 = (xc - w / 2) / 640 * original_width, (yc - h / 2) / 640 * original_height
            x2, y2 = (xc + w / 2) / 640 * original_width, (yc + h / 2) / 640 * original_height
            mask = 1 / (1 + np.exp(-row[4 + len(class_names):].reshape(160, 160)))
            mask = (mask > 0.5).astype('uint8') * 255
            mask_x1, mask_y1 = round(x1 / original_width * 160), round(y1 / original_height * 160)
            mask_x2, mask_y2 = round(x2 / original_width * 160), round(y2 / original_height * 160)
            mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
            mask = np.array(Image.fromarray(mask, 'L').resize((round(x2 - x1), round(y2 - y1))))
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if bool(contours[0]):
                polygon = [[contour[0][0], contour[0][1]] for contour in contours[0][0]]

            prediction_values.append([x1, y1, x2, y2, label, prob, polygon])
            bbox_coords.append([x1, y1, x2, y2])
            anchor_scores.append(prob)

        keep_indices = cv2.dnn.NMSBoxes(
            bboxes=np.array(bbox_coords, dtype=np.float32),scores=np.array(anchor_scores, dtype=np.float32),
            score_threshold=0.25,nms_threshold=0.4)

        prediction_values.sort(key=lambda x: x[4], reverse=True)
        prediction_values = [prediction_values[i] for i in keep_indices]

        for value in prediction_values:
            [x1, y1, x2, y2, label, prob, polygon] = value
            polygon = [(int(x1 + point[0]), int(y1 + point[1])) for point in polygon]

            draw.polygon(polygon, fill=(0, 255, 0, 100))
            draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
            draw.rectangle(draw.textbbox((x1 + 5, y1 - 5), f"{label}: {prob:.2f}", font=font), fill="red")
            draw.text((x1 + 5, y1 - 5), f"{label}: {prob:.2f}", fill="white", font=font)

        original_image.show()

batched_data = []
for idx,img_path in enumerate(data_set,1):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size)) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    batched_data.append(image)

    if len(batched_data) == batch_size:
        predict(batched_data,idx)
        batched_data = []
        
if batched_data:
    predict(batched_data,idx+idx%batch_size)