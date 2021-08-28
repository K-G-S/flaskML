import torch
import os
from pathlib import Path
from utils.general import check_requirements
from models.experimental import attempt_load
from utils.torch_utils import load_classifier, select_device

p = Path(__file__).parents[1]

def load_roi_model(weights=os.path.join(str(p), "yolov5/models_last/last-roi.pt"), device='', half=False):
    # Load model
    model = None
    modelc = None
    session = None
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    return classify, pt, onnx, stride, names, model, modelc, session, device


def load_digitrec_model(weights=os.path.join(str(p), "yolov5/models_last/last-digitrec.pt"), device='', half=False):
    # Load model
    model = None
    modelc = None
    session = None
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    return classify, pt, onnx, stride, names, model, modelc, session, device