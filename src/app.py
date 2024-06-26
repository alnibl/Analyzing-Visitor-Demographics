import cv2
from flask import Flask, Response
import os
import ultralytics
from ultralytics import YOLO
import numpy as np
import cv2
import time
from omegaconf import OmegaConf


config = OmegaConf.load('config/config.yml')
MODEL = config["model_path"]
model = YOLO(MODEL, task='detect')
imgsz = (224, 384)
SOURCE_VIDEO_PATH = config["source_video_path"]
print(model)
print(SOURCE_VIDEO_PATH)

app = Flask(__name__)
model_new = YOLO(MODEL, task='detect')

def model_detect_track_run(frame):

        img_ = frame.copy()

        start_time = time.time()
        results = model_new.track(source=frame, conf=0.01, imgsz=imgsz, save=False)

        #boxes = results[0].boxes.xywh.cpu()
        #track_ids = results[0].boxes.id.int().cpu().tolist()

        thumbnail_size = (60, 60)  # размер миниатюр
        thumbnail_offset = 10  # отступ между миниатюрами

        # Начальные координаты для размещения миниатюр
        start_x = frame.shape[1] - thumbnail_size[0] - thumbnail_offset
        start_y = thumbnail_offset

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            if boxes is not None and boxes.id is not None:
                bboxes_ = boxes.xyxy.tolist()
                bboxes = list(map(lambda x: list(map(lambda y: int(y), x)), bboxes_))
                # на будущее нужно обработать случай, когда сбивается трэк
                track_ids = boxes.id.int().cpu().tolist()

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
                    cropped_img = frame[y1:y2, x1:x2]
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
                #cv2.imwrite('data/1_.jpg', img_)
                print(f"Прошло: {round(time.time() - start_time, 3)} сек.")
                return img_
            else:
                print(f"Прошло: {round(time.time() - start_time, 3)} сек.")
                return None

def generate_frames():
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)  # Индекс 0 для первой камеры
    if not cap.isOpened():
        raise Exception("Could not open video device")

    while True:
        success, frame = cap.read()  # Чтение кадра
        if not success:
            break
        else:
            frame_ = model_detect_track_run(frame)# тут детекция, трекинг и обработка
            if frame_ is not None:
                frame = frame_
            ret, buffer = cv2.imencode('.jpg', frame)  # Кодирование кадра в JPEG
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)