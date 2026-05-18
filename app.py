import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn

MODEL_PATH = "bean_disease_model.pth"
IMAGE_SIZE = 128
classes = ["Healthy", "Angular Leaf Spot", "Bean Rust"]

model = models.googlenet(weights=None, aux_logits=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

st.title("Bean Leaf Disease Classifier")
st.write("Upload a bean leaf image to predict disease category.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_class = classes[prediction.item()]
    confidence_score = confidence.item() * 100
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    with st.expander("See raw model output"):
            st.write("Probabilities:", probabilities.numpy())
            st.write("Class Index:", prediction.item())