import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F

class ImageDatasetFromFolders(Dataset):
    def __init__(self, path, transform=ToTensor):
        self.path = path
        self.class_names = sorted(os.listdir(self.path))
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
    
base = os.getcwd()
train_dir = os.path.join(base,'Train')
test_dir = os.path.join(base,'Test')
val_dir = os.path.join(base,'Validation')

image_size = (256,256)
transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor()])
#transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize(image_size)])
# transform = transforms.Compose([transforms.Resize(image_size)])
train_data = ImageDatasetFromFolders(train_dir, transform=transform)
test_data = ImageDatasetFromFolders(test_dir, transform=transform)
val_data = ImageDatasetFromFolders(val_dir, transform=transform)
print((train_data[2][0]).dtype) #check the type of the data to make sure it is torch.float32
print(train_data[1000][0]) #check if the data has been scaled
print(test_data.__len__())
print(val_data.__len__())

batch_size = 32
train_dl = DataLoader(train_data, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size, shuffle=True)
val_dl = DataLoader(val_data, batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=None)
        # 2nd convolutional layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)   # your input size
            out = self._forward_features(dummy)
            flat_dim = out.numel()

        # Fully connected layer
        self.fc1 = nn.Linear(flat_dim, num_classes)
#         self.fc2 = nn.Linear(512,128)
#         self.fc3 = nn.Linear(128,num_classes)
        

    def _forward_features(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
           x: Input tensor.

        Returns:
           torch.Tensor
               The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.dropout(x, p=0.2, training=self.training) 
        x = F.relu(self.conv3(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv4(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)
        x = F.dropout(x, p=0.2, training=self.training)  
        return x
    
    def forward(self,x):
        x = self._forward_features(x)
#         x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = torch.flatten(x,1)
#         print(x.shape)
        x = self.fc1(x)            # Apply fully connected layer
#         x = F.dropout(x, p=0.4, training=self.training) 
#         x = self.fc2(x)
#         x = self.fc3(x)
        return x

model = CNN(in_channels=3, num_classes=3)
print(model)

#from torchinfo import summary
#print(summary(model, (32,3,256,256)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train_model(model, train_loader, val_loader, criterion, optimizer,model_name, num_epochs=10):
    print(f'Starting training the {model_name}.pt.......')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    train_accuracy = []
    train_loss = []
    validation_accuracy = []
    validation_loss = []
    num_epochs = num_epochs
    for epoch in range(1, num_epochs+1):
        print(f"Epoch [{epoch}/{num_epochs}]")
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item() * images.size(0)  # loss.item() is the avg loss per batch
            #_, predicted = torch.max(out.data, 1)
            #total_train += labels.size(0)
            acc_train_batch = (out.argmax(1) == labels).float().mean()
            correct_train += acc_train_batch.item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / len(train_loader)
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_acc)

        model.eval()  # Set model to evaluation mode (disables dropout, uses running stats for batch norm)
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                out = model(images)
                loss = criterion(out, labels)

                val_loss += loss.item() * images.size(0)
                #_, predicted = torch.max(out.data, 1)
                #total_val += labels.size(0)
                acc_val_batch = (out.argmax(1) == labels).float().mean()
                correct_val += acc_val_batch.item()

            # Calculate validation statistics for the epoch
        epoch_val_loss = val_loss / len(val_loader.dataset)  
        epoch_val_acc = correct_val / len(val_loader)          
        validation_loss.append(epoch_val_loss)                    
        validation_accuracy.append(epoch_val_acc) 


        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {epoch_val_acc:.4f}")
        print('-'*30)
        
    print("Finished Training.....")
    print('Saving model and the training accuracies....')
    torch.save(model, f'{model_name}.pt')
    training_details = {'train_losses': train_loss, 'train_accuracies': train_accuracy, 
                        'val_losses': validation_loss, 'val_accuracies': validation_accuracy}
    torch.save(training_details, f'train_val_accuracy_data.pt')
    return training_details

train_model(model, val_dl, val_dl, criterion, optimizer,'recognize_disease_25_12')
