import gradio as gr
from models.model import PotentialModel
import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from models.model import PotentialModel
from lib.model_lib import (PearsonR, config_parser, load_state_dict, generatecontcar, findinistru, eneforlenoutput)
from default_parameters import  default_train_config, default_data_config



test_config = {**default_data_config, **default_train_config}


model = PotentialModel(test_config['gat_node_dim_list'],
                               test_config['energy_readout_node_list'],
                               test_config['force_readout_node_list'],
                               test_config['head_list'],
                               test_config['bias'],
                               test_config['negative_slope'],
                               test_config['device'],
                               test_config['tail_readout_no_act'])
optimizer = optim.AdamW(model.parameters(),
                      lr=test_config['learning_rate'],
                      weight_decay=test_config['weight_decay'])

print(model)

# load stat dict if there exists.
if os.path.exists(os.path.join(test_config['model_save_dir'], 'agat_state_dict.pth')):
    try:
        print(os.path.abspath(os.getcwd()))
        print(test_config['model_save_dir'])
        checkpoint = load_state_dict(test_config['model_save_dir'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(test_config['device'])
        model.device = test_config['device']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(
            f'User info: Model and optimizer state dict loaded successfully from {self.test_config["model_save_dir"]}.')
    except:
        print('User warning: Exception catched when loading models and optimizer state dict.')
else:
    print('User info: Checkpoint not detected')


# predictor = Predictone()

def generate_and_model(molecularname):
    return molecularname, molecularname


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Machine Learning for Accelerated Peak Force Prediction")
    gr.Markdown("**We introduce CBPFNet (a pretrained Covalent Bond Peak Force Network), "
                "an advanced graph attention network model that simulates the dynamic process of covalent bond cleavage and accurately "
                "predicts stress responses under various mechanical loads. Based on the trained CBPFNet, "
                "we have developed an automated program capable of predicting the strength of covalent bonds in organic molecules "
                "using the optimized structure file, CONTCAR.**")
    with gr.Row():
        with gr.Column():
            gr.Image("architecture.jpg", label="The architecture of CBPFNet",
                     width=600, height=500)
        with gr.Column():
            gr.Image("predataset.jpg", label="The model performance of Pre-trained and Well-trained",
                     width=600, height=500)

    with gr.Row():
        gr.Markdown("**Choose the desired molecular.**")

    with gr.Row():
        with gr.Column(scale=3):
            molecularname = gr.Dropdown(["cat", "dog", "bird"], label="Organic Molecular", info="Will add more animals later!")
            generate_button = gr.Button("Calculate")
            contcartxt = gr.Textbox(label="CONTCAR File")

        with gr.Column(scale=3):
            predictforce = gr.Textbox(label="Predicted peak force")
            predictplot = gr.Image(label="Prediction Curve")

    generate_button.click(
        generate_and_model,
        inputs=[molecularname],
        outputs=[predictforce, contcartxt]
    )

demo.launch(share=True)