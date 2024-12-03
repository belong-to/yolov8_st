from pathlib import Path

# Source
SOURCES_LIST = ["图片", "视频", "摄像头"]

# DL model config
DETECTION_MODEL_DIR = Path('weights', 'detection')
INSTANCE_SEGMENTATION_MODEL_DIR = Path('weights', 'instance_segmentation')
CLASSIFICATION_MODEL_DIR = Path('weights', 'classification')
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"

DETECTION_MODEL_LIST = [
    "目标检测.pt","实例分割.pt","图像分类.pt"]

INSTANCE_SEGMENTATION_MODEL_LIST = [
     "目标检测.pt","实例分割.pt","图像分类.pt"]
DEFAULT_INSTANCE_SEGMENTATION_MODEL = "实例分割.pt"

CLASSIFICATION_MODEL_LIST = [
    "目标检测.pt","实例分割.pt","图像分类.pt"]
DEFAULT_CLASSIFICATION_MODEL = "图像分类.pt"