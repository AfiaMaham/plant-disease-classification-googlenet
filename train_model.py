
import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import os

# =========================
# CONFIG
# =========================
DATASET_PATH = "bean-leaf-lesions-classification"
CSV_PATH = os.path.join(DATASET_PATH, "train.csv")
MODEL_SAVE_PATH = "bean_disease_model.pth"
LABEL_SAVE_PATH = "classes.txt"

LR = 1e-3
BATCH_SIZE = 8
EPOCHS = 10
IMAGE_SIZE = 128

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

# Create full image paths
full_paths = []
for path in df["image:FILE"]:
    full_paths.append(os.path.join(DATASET_PATH, path))

df["image_path"] = full_paths

# Train-Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["category"]
)

# Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(df["category"])

# Save classes
with open(LABEL_SAVE_PATH, "w") as f:
    for cls in df["category"].unique():
        f.write(str(cls) + "\n")

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================
# DATASET CLASS
# =========================
class BeanDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = torch.tensor(
            label_encoder.transform(dataframe['category'])
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# =========================
# DATALOADERS
# =========================
train_dataset = BeanDataset(train_df, transform)
test_dataset = BeanDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL
# =========================
model = models.googlenet(weights='DEFAULT')

num_classes = len(label_encoder.classes_)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# =========================
# TRAINING
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

print("Training started...\n")

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    total_correct = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (
            torch.argmax(outputs, dim=1) == labels
        ).sum().item()

    accuracy = (total_correct / len(train_dataset)) * 100

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

# =========================
# TESTING
# =========================
model.eval()
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()

accuracy = (correct / len(test_dataset)) * 100

print(f"\nTest Accuracy: {accuracy:.2f}%")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\nModel saved as: {MODEL_SAVE_PATH}")


