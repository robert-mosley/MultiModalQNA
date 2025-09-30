import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch.nn.functional as F
from torchvision import transforms
import math

text_model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

context = load_dataset("coco_captions")
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

class modelCNN(nn.Module):
    def __init__(self):
        super(modelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2,2)
        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = self.pool2(x)
        x = x.view(-1, 512)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class MultiModel(nn.Module):
    def __init__(self, cnn_model, text_model, embed_dim):
        super(MultiModel, self).__init__()
        self.cnn_model = cnn_model
        self.text_encoder = text_model
        self.cnn_proj = nn.Linear(768, embed_dim)
        self.text_proj = nn.Linear(768, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))
    def forward(self, img, text_inputs, attn_mask):
        img = self.cnn_model(img)
        img_features = self.cnn_proj(img)
        txt_outputs = self.text_encoder(input_ids=text_inputs,
                                        attention_mask=attn_mask,
                                        output_hidden_states=True)
        text_features = txt_outputs.last_hidden_state[:,0,:]
        img_features = F.normalize(img_features, dim=-1)
        text_features = F.normalize(self.text_proj(text_features), dim=-1)
        logit_scale = self.logit_scale.exp()
        scores = logit_scale * (img_features @ text_features.T)
        return scores

class CreateDataset(Dataset):
    def __init__(self, dataset, transform=transform, tokenizer=tokenizer):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx]['image'])
        caption = self.dataset[idx]['caption'][0]
        text = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        return image, text

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    input_ids = torch.cat([item[1]['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item[1]['attention_mask'] for item in batch], dim=0)
    return images, input_ids, attention_mask

data = CreateDataset(context["train"])
dataset = DataLoader(data, batch_size=8, shuffle=True, collate_fn=collate_fn)
TIModel = MultiModel(modelCNN(), text_model, 256)
optimizer = torch.optim.AdamW(TIModel.parameters(), lr=1e-4)

def loss(logits):
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    return (loss1 + loss2) / 2

def train(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIModel.to(device)
    TIModel.train()
    for epoch in range(epochs):
        for images, input_ids, attention_mask in dataset:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            pred = TIModel(images, input_ids, attention_mask)
            batch_loss = loss(pred)
            batch_loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, loss: {batch_loss.item()}")

train(5)
