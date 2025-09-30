# 🖼️ Multimodal Image Captioning & Question Answering

This project is a **multimodal AI system** that can **generate captions for images** and **answer natural language questions about them**.  
It combines **computer vision (BLIP)** with **natural language processing (DistilBERT + LoRA fine-tuning)** for interactive image understanding.  

In addition, I **implemented my own version of CLIP (Contrastive Language-Image Pretraining)** from the original research paper.  
While the custom CLIP implementation is not directly used in this system, it demonstrates my ability to **translate research into working code** and deepens understanding of **multimodal embeddings and contrastive learning**.

---

## ✨ Features
- 📷 **Image Captioning** – Generate captions for single images or all images in a folder using [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base).  
- ❓ **Visual Question Answering (VQA)** – Answer questions about an image using a **LoRA fine-tuned DistilBERT** model.  
- 🔎 **Folder Search** – Generate captions for an entire folder of images and search for matches with a given keyword.  
- 🖥️ **Interactive GUI** – Simple file/folder selection powered by Tkinter.  
- 📊 **Visualization** – Matplotlib displays images with their generated captions.  
- 🧪 **Research Exploration** – Custom CLIP implementation to explore cross-modal contrastive learning.  

---

## 🛠️ Tech Stack
- **Python**  
- **PyTorch**  
- **Hugging Face Transformers**  
- **PEFT (LoRA fine-tuning)**  
- **BLIP (Salesforce)**  
- **DistilBERT**  
- **Custom CLIP (from research paper)**  
- **Tkinter, Matplotlib, PIL**  

---

## 🚀 Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/robert-mosley/MultiModalQNA.git
   cd MultiModalQNA
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Make sure you have a LoRA adapter checkpoint saved locally.
Update the adapter_path in the script to point to your checkpoint:

python
Copy code
adapter_path = "../checkpoint-11066"
▶️ Usage
1. Folder Captioning + Search
Generate captions for all images in a folder and search for a keyword:

bash
Copy code
python main.py 1 <keyword>
Select a folder when prompted.

Captions for each image are generated automatically.

Images containing the <keyword> in their captions will be displayed.

2. Single Image Question Answering
Ask a question about a single image:

bash
Copy code
python main.py 2 "<your question>"
Select an image file when prompted.

The program will generate a caption and attempt to answer your question.

If no clear answer is found, the caption is printed instead.

📂 Project Structure
pgsql
Copy code
multimodal-vqa/
│── main.py               # Core script (captioning + Q&A)
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── checkpoint-11066/     # LoRA adapter files (not included in repo)
│── clip-from-scratch/    # Custom CLIP implementation (research exploration)