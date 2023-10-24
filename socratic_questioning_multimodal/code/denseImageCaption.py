# %%
# image caption
# image generation (stable difusion)
# grounded caption
# image grounding
# vqa
# object detection


# %%
import torch
from transformers import pipeline

import multiprocessing as mp
import os
import time
import sys
import glob
import json
import numpy as np
import random

from PIL import Image
from lavis.models import load_model_and_preprocess

from transformers import BlipProcessor, BlipForConditionalGeneration

import openai
import pdb

import os
# %%
class DenseImageCaption:
    def __init__(self, api,gpu_id, llm=None):
        os.environ['CUDA_VISIBLE_DEVICES']= str(gpu_id)
        torch.cuda.set_device(gpu_id)
        self.device = torch.device('cuda:'+str(gpu_id))
        
        print('Load: blip2 model')
        # blip2 model        
        self.blip2model, self.blip2vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=self.device
        )
        
        self.llm = None
        if llm is None:
            openai.api_key = api[0]
        else:
            self.llm = llm
    
    def blip2caption(self, question, img_path):
        raw_image = Image.open(img_path).convert('RGB')
        device = self.device
        image = self.blip2vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        caption = self.blip2model.generate({"image": image, "prompt": "Generate a caption about \"" + question + "\""})
        
        return caption

    def get_captionOnly(self, question, img_path):
        caption = self.blip2caption(question, img_path)
        return caption[0].strip()
        
    def get_visual_descrip(self, question, img_path): 
        caption = self.get_captionOnly(question, img_path)
        img_summary = "Image Caption: " + caption + "\n"
        return img_summary
