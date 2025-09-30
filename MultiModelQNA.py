from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel, PeftConfig
from PIL import Image
import torch
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.image as mping
import matplotlib.pyplot as plt
import os
import sys

adapter_path = "../checkpoint-11066"

if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
    raise FileNotFoundError(f"LoRA adapter not found at: {adapter_path}")

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
model2 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
adapterModel = PeftModel.from_pretrained(model, adapter_path, adapter_file_name="adapter_model.safetensors")
adapterModel.eval()

option = sys.argv[1]
if option == "1":
    captions = []

    context = filedialog.askdirectory(
        title="select folder"
    )

    files = os.listdir(context)
    def gen_caption(img):
        path = os.path.join(context, img)
        print(path)
        image = Image.open(path).convert("RGB")
        features = processor(images=image, return_tensors="pt")
        caption = model2.generate(**features)
        img = processor.decode(caption[0], skip_special_tokens=True)
        return img

    for f in files:
        captions.append(gen_caption(f))

    word = sys.argv[2]
    for cap in captions:
        if word in cap:
            img = mping.imread(os.path.join(context, files[captions.index(cap)]))
            plt.title(cap)
            plt.imshow(img)
            plt.axis("off")
            plt.show()

if option == "2":
    context = filedialog.askopenfilename(
        title="Select File"
    )
    context = Image.open(context).convert("RGB")
    features = processor(images=context, return_tensors="pt")
    out = model2.generate(**features)
    caption = processor.decode(out[0], skip_special_tokens=True)

    question = sys.argv[2]

    inputs = tokenizer(question, caption, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = adapterModel(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    )
    if answer == "":
        print(caption)
    else:
        print("Answer:", answer)