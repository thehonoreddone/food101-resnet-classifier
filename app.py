import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision import transforms,datasets,models

dataset = datasets.Food101(root="data", split="train", download=False)

class_names = dataset.classes

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    model.load_state_dict(torch.load("model0.pth", map_location="cpu"))
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),         
    transforms.ToTensor(),                
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



st.title("🍕 Food101 Görüntü Sınıflandırma")

upload_file = st.file_uploader("Yiyecek fotoğrafı yükle", type=["jpg", "png", "jpeg"])

if upload_file:
    image = Image.open(upload_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    model = load_model()

    image_tensor = transform(image).unsqueeze(0)

    try:
        with torch.no_grad():
            preds = model(image_tensor)
            probs = torch.softmax(preds, dim=1)
            pred_class = class_names[probs.argmax(dim=1).item()]
            pred_prob = probs.max().item()

        st.markdown(f"**Tahmin:** `{pred_class}` ({pred_prob:.2%} olasılık)")

    except Exception as e:
        st.error(f"Tahmin başarısız oldu: {e}")
