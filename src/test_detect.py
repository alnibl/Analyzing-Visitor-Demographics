import os
import ultralytics
from ultralytics import YOLO
import numpy as np
import cv2
import time
from omegaconf import OmegaConf
print("ultralytics.__version__:", ultralytics.__version__)

HOME = os.getcwd()
print(HOME)

config = OmegaConf.load('config/config.yml')

MODEL = config["model_path"]
model = YOLO(MODEL, task='detect')
print(model)

imgsz = (224, 384)

SOURCE_VIDEO_PATH = config["source_video_path"]
print(SOURCE_VIDEO_PATH)
'''
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Всего кадров: {fcount}, Продолжительность: {round(fcount/fps, 2)} сек.")

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

if not cap.isOpened():
    print("Не удалось открыть видеофайл.")

TARGET_VIDEO_PATH = config["target_video_path"]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_fps = int(fps/2)
print(f"fps камеры: {fps}, width: {width}, height: {height}")

out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, target_fps, (width, height))
'''

img = cv2.imread(SOURCE_VIDEO_PATH)
img_ = img.copy()
model_new = YOLO(MODEL, task='detect')

start_time = time.time()
results = model_new.track(source=frame, conf=0.01, imgsz=(224, 384), save=False)

#boxes = results[0].boxes.xywh.cpu()
#track_ids = results[0].boxes.id.int().cpu().tolist()

thumbnail_size = (60, 60)  # размер миниатюр
thumbnail_offset = 10  # отступ между миниатюрами

# Начальные координаты для размещения миниатюр
start_x = img.shape[1] - thumbnail_size[0] - thumbnail_offset
start_y = thumbnail_offset

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    bboxes_ = boxes.xyxy.tolist()
    bboxes = list(map(lambda x: list(map(lambda y: int(y), x)), bboxes_))
    track_ids = results[0].boxes.id.int().cpu().tolist()

    confs_ = boxes.conf.tolist()
    confs = list(map(lambda x: int(x * 100), confs_))
    classes_ = boxes.cls.tolist()
    classes = list(map(lambda x: int(x), classes_))
    cls_dict = result.names
    class_names = list(map(lambda x: cls_dict[x], classes))

    print(f"bboxes: {bboxes}")
    print(f"track_ids: {track_ids}")
    print(f"class_names: {class_names}, confs: {confs}")

    for bbox, conf, class_name, id in zip(bboxes, confs, class_names, track_ids):
        x1, y1, x2, y2 = bbox
        # Отрисовка bounding box
        cv2.rectangle(img_, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"id {id}, {class_name} {conf}"
        # Отрисовка текста
        cv2.putText(img_, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Вырезание части изображения, которую обводит bounding box
        cropped_img = img[y1:y2, x1:x2]
        # Уменьшение размера вырезанной части до миниатюры
        thumbnail = cv2.resize(cropped_img, thumbnail_size)

        # Определение координат для размещения миниатюры
        thumbnail_y1 = start_y
        thumbnail_y2 = start_y + thumbnail_size[1]
        thumbnail_x1 = start_x
        thumbnail_x2 = start_x + thumbnail_size[0]

        # Размещение миниатюры на оригинальном изображении
        img_[thumbnail_y1:thumbnail_y2, thumbnail_x1:thumbnail_x2] = thumbnail

        # Смещение для следующей миниатюры
        start_y += thumbnail_size[1] + thumbnail_offset
# сохранить кадр + миниатюра
cv2.imwrite('data/1_.jpg', img_)

print(f"Прошло: {round(time.time() - start_time, 3)} сек.")