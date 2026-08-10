import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

base = os.getcwd()

train_dir = os.path.join(base, "Train")
val_dir = os.path.join(base, "Validation")
test_dir = os.path.join(base, "Test")

# lower resolution for now - testing purposes
image_size = (64, 64)
batch_size = 32

transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])


class ImageDatasetFromFolders(Dataset):
    def __init__(self, path, transform=ToTensor):
        self.path = path
        self.class_names = ["Healthy", "Powdery", "Rust"]
        self.transform = transform
        complete_dataset = []
        for dir in os.listdir(self.path):
            for image in os.listdir(os.path.join(self.path, dir)):
                complete_dataset.append((dir, os.path.join(self.path, dir, image)))
        self.complete_dataset = complete_dataset
        
    def __len__(self):
        return len(self.complete_dataset)
    
    def get_example(self, idx):
        img_path = self.complete_dataset[idx][1]
        image = Image.open(img_path)
        label = self.complete_dataset[idx][0]
        label = self.class_names.index(label)
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def scale_image(self, img):
        max_pixel = np.max(np.array(img))
        img = img/max_pixel
        return img
    
    def __getitem__(self, idx):
        img_path = self.complete_dataset[idx][1]
        image = Image.open(img_path)
        label = self.complete_dataset[idx][0]
#         print(label)
        label = self.class_names.index(label)
#         print(label)
        if self.transform:
            image = self.transform(image)
#            image = self.scale_image(image)
        return image, label

train_data = ImageDatasetFromFolders(train_dir, transform)
val_data = ImageDatasetFromFolders(val_dir, transform)
test_data = ImageDatasetFromFolders(test_dir, transform)

print("Classes:", train_data.class_names)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Test images:", len(test_data))

assert (train_data.class_names == val_data.class_names == test_data.class_names)

train_dl = DataLoader(train_data,batch_size=batch_size,shuffle=True, seed)

val_dl = DataLoader(val_data,batch_size=batch_size,shuffle=False)

test_dl = DataLoader(test_data,batch_size=batch_size,shuffle=False)

# --------------------------------------------------
# 3. Fully connected neural network
# --------------------------------------------------

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        input_features = 3 * 64 * 64

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Return logits, not softmax probabilities.
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


model = NeuralNetwork(num_classes=3).to(device)

print(model)

num_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable parameters: {num_params:,}")

# --------------------------------------------------
# 4. Loss and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=5e-4
)

# --------------------------------------------------
# 5. Training and validation
# --------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=20
):
    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
    }

    model.to(device)

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += (
                    loss.item() * images.size(0)
                )

                val_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

    return history


history = train_model(
    model,
    train_dl,
    val_dl,
    criterion,
    optimizer,
    num_epochs=20
)

# --------------------------------------------------
# 6. Save model and training history
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

torch.save(
    model.state_dict(),
    "results/pytorch_nn_state_dict.pth"
)

torch.save(
    history,
    "results/pytorch_nn_history.pt"
)

# --------------------------------------------------
# 7. Plot training and validation curves
# --------------------------------------------------

epochs = range(1, len(history["train_losses"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(
    epochs, history["train_losses"],
    label="Training loss"
)
axes[0].plot(
    epochs, history["val_losses"],
    label="Validation loss"
)
axes[0].set_title("NN: Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(
    epochs, history["train_accuracies"],
    label="Training accuracy"
)
axes[1].plot(
    epochs, history["val_accuracies"],
    label="Validation accuracy"
)
axes[1].set_title("NN: Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/pytorch_nn_curves.png", dpi=150)
plt.show()

# --------------------------------------------------
# 8. Evaluate on test data
# --------------------------------------------------

model.eval()

all_predictions = []
all_labels = []

test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_dl:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_labels.extend(
            labels.cpu().numpy()
        )

test_loss /= test_total
test_accuracy = test_correct / test_total

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        target_names=train_data.class_names,
        zero_division=0
    )
)

# --------------------------------------------------
# 9. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_data.class_names
)

disp.plot(cmap="Blues", values_format="d")
plt.title("PyTorch NN - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "results/pytorch_nn_confusion_matrix.png",
    dpi=150
)
plt.show()
