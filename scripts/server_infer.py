


import os
import sys
from pathlib import Path
# 保证可以找到 genie_sim_ros 和 lerobot
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import threading

import cv2
import numpy as np
import torch

import draccus
from dataclasses import dataclass

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import argparse
import base64
import cv2
from flask_socketio import SocketIO, emit
from typing import Union

from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel

@dataclass
class GenerateConfig:

    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/finetuned"

    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False  # Center crop? (if trained w/ random crop image aug)
    local_log_dir: str = "./experiments/eval_logs"  # Local directory for eval logs
    seed: int = 7

    action_decoder_path: str = "checkpoints/finetuned/action_decoder.pt"
    window_size: int = 30

    n_layers: int = 2
    hidden_dim: int = 1024

    with_proprio: bool = True

    smooth: bool = False
    balancing_factor: float = 0.1  # larger for smoother

    task_name: str = "iros_stamp_the_seal"

def get_instruction(task_name):

    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")

    return lang

def make_policy():
    cfg = GenerateConfig()
    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    policy = WrappedGenieEvaluation(cfg, wrapped_model)
    return policy, cfg

policy, _ = make_policy()
    
def parse_args():
    parser = argparse.ArgumentParser(description='VLNCE 4-class server')
    # parser.add_argument('--config', type=str, default='/work/configs/250506_vlnce_4cls_benchmark.yaml', help='Path to the config file')
    # parser.add_argument('--model_path', type=str, default='/work/outputs/epoch-02-loss=0.2527.pt', help='Path to the model checkpoint')
    return parser.parse_args()

args = parse_args()
def preprocess_img(img, device='cuda'):
    chw_img = img.transpose(2, 0, 1)
    chw_img = torch.from_numpy(chw_img / 255.0).float().unsqueeze(0).to(device)
    return chw_img

def preprocess_state(state, device='cuda'):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    return state_tensor

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*") 
########################################
@socketio.on('select_action')
def process(data):
    try:
        state = np.array(data['state'])
        command = data['command']  # 你可以根据 command 动态生成 lang

        head_img_data = base64.b64decode(data['image_head'])
        head_img_array = np.frombuffer(head_img_data, np.uint8)
        head_img = cv2.imdecode(head_img_array, cv2.IMREAD_COLOR)

        left_img_data = base64.b64decode(data['image_hand_left'])
        left_img_array = np.frombuffer(left_img_data, np.uint8)
        left_img = cv2.imdecode(left_img_array, cv2.IMREAD_COLOR)

        right_img_data = base64.b64decode(data['image_hand_right'])
        right_img_array = np.frombuffer(right_img_data, np.uint8)
        right_img = cv2.imdecode(right_img_array, cv2.IMREAD_COLOR)

        lang = get_instruction(command)  # 使用 genie 的函数来获取 lang 指令

        # if cfg.with_proprio:
        print("head_img shape:", head_img.shape)
        print("left_img shape:", left_img.shape)
        print("right_img shape:", right_img.shape)
        print("lang:", lang)
        print("state.shape:", state.shape)
        action = policy.step(head_img, left_img, right_img, lang, state)
        # else:
            # action = policy.step(head_img, left_img, right_img, lang)

        return {"action": action.tolist()}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"response": "Server error: " + str(e)}


@app.route('/')
def index():
    return "VLNCE 4-class server is running!"

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)