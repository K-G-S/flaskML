import torch
import os
from pathlib import Path
from utils.general import check_requirements
from models.experimental import attempt_load
from utils.torch_utils import select_device

from models.common import DetectMultiBackend

p = Path(__file__).parents[1]

def pre_load_model(weights="yolov5/models_last/last-digitrec.pt", device='', half=False, dnn=False):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

    return  stride, names, pt, jit, onnx, engine, model, device