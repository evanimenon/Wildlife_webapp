import os
import sys
from pathlib import Path
import io

import torch
import torch.backends.cudnn as cudnn
from pymongo import MongoClient
from bson.objectid import ObjectId
import gridfs
from PIL import Image
import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from tag_images import categorize_images  # Import the function

@torch.no_grad()
def run(
        weights='runs/train/wii_28_072/weights/best.pt',
        data='data/wii_aite_2022_testing.yaml',
        imgsz=(640, 640),
        conf_thres=0.001,
        iou_thres=0.6,
        max_det=1000,
        device='0' if torch.cuda.is_available() else 'cpu',
        view_img=False,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        project='runs/detect',
        name='yolo_test_24_08_site0001',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        mongodb_uri='mongodb+srv://kushiluv:kushiluv25@cluster0.pety1ki.mongodb.net/',
        file_ids='6693c174da65f3418db3c749'
):
    for i in range(torch.cuda.device_count()):
        print("in")
        print(torch.cuda.get_device_name(i))
    if not mongodb_uri:
        raise ValueError("MongoDB URI must be provided")
    if not file_ids:
        raise ValueError("File IDs must be provided")
    
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client['ImageDatabase']
    fs = gridfs.GridFS(db)
    
    file_ids = file_ids.split(',')
    images = []
    
    for file_id in file_ids:
        image_doc = fs.get(ObjectId(file_id))
        image = Image.open(io.BytesIO(image_doc.read()))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        images.append((image, file_id))
    
    if len(images) == 0:
        raise ValueError("No images found in MongoDB")

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    seen, dt = 0, [0.0, 0.0, 0.0]
    output_file_ids = []
    label_texts = {}

    for im0, file_id in images:
        im = cv2.resize(im0, (imgsz[1], imgsz[0]))
        im = im.transpose(2, 0, 1)  # HWC to CHW
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im.unsqueeze(0)  # expand for batch dim
        t1 = time_sync()

        # Inference
        visualize = False  # disable visualization for now
        pred = model(im, augment=augment, visualize=visualize)
        t2 = time_sync()
        dt[1] += t2 - t1

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t2

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = file_id, im0.copy(), 0
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            txt_output = io.StringIO()
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # Rescale boxes from img_size to im0 size

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        txt_output.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if not nosave:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=Path(f'/tmp/{Path(p).stem}_crop.jpg'), BGR=True)

            # Save txt content to dictionary
            if save_txt or save_conf:
                label_texts[str(file_id)] = txt_output.getvalue()

            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if not nosave:
                _, buffer = cv2.imencode('.jpg', im0)
                output_file_id = fs.put(buffer.tobytes(), filename=f"processed_{file_id}.jpg", metadata={"original_id": str(file_id)})
                output_file_ids.append(str(output_file_id))

        LOGGER.info(f'Done. ({t2 - t1:.3f}s)')

    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image

    # Print output file IDs
    print("Output File IDs:", output_file_ids)
    print("Label Texts:", label_texts)

    # Call categorize_images directly
    categorize_images(mongodb_uri, ','.join(output_file_ids), str(label_texts))

if __name__ == "__main__":
    # Default values for the parameters
    weights = 'runs/train/wii_28_072/weights/best.pt'
    data = 'data/wii_aite_2022_testing.yaml'
    imgsz = (640, 640)
    conf_thres = 0.001
    iou_thres = 0.6
    max_det = 1000
    device = '0' if torch.cuda.is_available() else 'cpu'
    view_img = False
    save_txt = True
    save_conf = True
    save_crop = False
    nosave = False
    classes = None
    agnostic_nms = False
    augment = False
    visualize = False
    project = 'runs/detect'
    name = 'yolo_test_24_08_site0001'
    exist_ok = False
    line_thickness = 3
    hide_labels = False
    hide_conf = False
    half = False
    dnn = False
    mongodb_uri = 'mongodb+srv://kushiluv:kushiluv25@cluster0.pety1ki.mongodb.net/'
    file_ids = "6693d7d918ef25cf577f6b33,6693d7d918ef25cf577f6b35,6693d7d918ef25cf577f6b37"

    run(weights=weights, data=data, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
        device=device, view_img=view_img, save_txt=save_txt, save_conf=save_conf, save_crop=save_crop,
        nosave=nosave, classes=classes, agnostic_nms=agnostic_nms, augment=augment, visualize=visualize,
        project=project, name=name, exist_ok=exist_ok, line_thickness=line_thickness, hide_labels=hide_labels,
        hide_conf=hide_conf, half=half, dnn=dnn, mongodb_uri=mongodb_uri, file_ids=file_ids)
