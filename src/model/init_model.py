import torch
import os

from model import def_model

def init():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gaze_predictor = def_model.gaze_network_v2()
    gaze_predictor.cuda()
    ckpt = torch.load(os.path.join(script_dir,"ckpt/best_model_3.26_0.29.pth"))
    gaze_predictor.load_state_dict(ckpt['model_state_dict'], strict=True)
    gaze_predictor.eval()
    print("init model successfully")
    return gaze_predictor
