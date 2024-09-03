from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from io import BytesIO

import base64
from PIL import Image
import nltk
from transformers import LayoutLMv2ForTokenClassification
from transformers import LayoutLMv2Processor
import os

import psutil
from transformers import AutoProcessor, AutoModelForTokenClassification
from datetime import datetime
import uvicorn
import requests
import socket
import json
import pycountry
from typing import Optional
from geonamescache import GeonamesCache
import re
import string
import numpy as np
from typing import Union, List

nltk.download('punkt')  # Download the necessary tokenization test_data
import torch
from copy import deepcopy
import torch
from PIL import Image
import torchvision.transforms.functional as F

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import time
import torch
from torchvision.transforms import functional as F
from transformers import LayoutLMv2ForTokenClassification
from transformers import LayoutLMv2Processor
import base64
from PIL import Image

model_path = "/datadrive/Trained_Models/Extraction/PO/Best_Model"
model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)
class LayoutLMv2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super(LayoutLMv2Wrapper, self).__init__()
            self.model = model

        def forward(self, input_ids, bbox, image, attention_mask, token_type_ids):
            outputs = self.model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return outputs.logits
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")  # Convert to RGB if necessary
    image_tensor = F.to_tensor(image)
    return image_tensor

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model_wrapper = LayoutLMv2Wrapper(model)
image_tensor = preprocess_image("/datadrive/rakesh/EVAL/BOL/Images/Bill_of_lading_3_page_0.png")
inputs = processor(image_tensor, return_tensors="pt", padding="max_length", truncation=True)
input_data = inputs.to("cpu")

traced_model = torch.jit.trace(model_wrapper,
                    example_inputs=[input_data['input_ids'].cpu(),
                input_data['bbox'].cpu(),
                                    input_data['image'].cpu(),
                                    input_data['attention_mask'].cpu(),
                                    input_data['token_type_ids'].cpu(),],
                    check_trace=False,strict=False)
torch.jit.save(traced_model,"/datadrive/rakesh/traced_models/traced_model_po.pt")
print("tracing done !!! for the llmv2 for extraction")