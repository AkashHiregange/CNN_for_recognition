import numpy as np 
import matplotlib.pyplot as plt
import torch
from time import perf_counter

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config, save_files=True, file_name='training', num_epochs=20):

    
    
    history = {"train_losses": [],"train_accuracies": [],"val_losses": [],"val_accuracies": []}

    model.to(device)

    config.create_directories()
    print('relevant directories created ... ')
    
    start = perf_counter()
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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
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

                val_running_loss += (loss.item() * images.size(0))

                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    end = perf_counter()
    elapsed_time = end-start
    print(f'\n Training time: {elapsed_time}')
    
    if save_files:
        torch.save(model.state_dict(),f"{config.model_dir}/{file_name}_state_dict.pth")
        torch.save(history,f"{config.model_dir}/{file_name}_history.pt")
    return history, elapsed_time

def test_model(model, test_loader, criterion, class_names, device, config, file_name='confusion_matrix', plot_confusion_matrix=True):
    from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
    
    model.eval()

    config.create_directories()
    print('relevant directories created ... ')

    all_predictions = []
    all_labels = []
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            test_loss += loss.item() * images.size(0)
    
            predictions = outputs.argmax(dim=1)
    
            test_correct += (predictions == labels).sum().item()
    
            test_total += labels.size(0)
    
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= test_total
    test_accuracy = test_correct / test_total
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print('--'*20)
    print('Classification Report')
    print('--'*20)
    print('\n')
    
    print(classification_report(all_labels,all_predictions,labels=[0,1, 2], target_names=class_names,zero_division=0))
    
    cm = confusion_matrix(all_labels,all_predictions,labels=[0, 1, 2])

    if plot_confusion_matrix:
        cm = confusion_matrix(all_labels,all_predictions,labels=[0, 1, 2])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
        
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{config.plot_dir}/{file_name}.png",dpi=150)
        plt.show()

    return test_loss, test_accuracy, all_predictions, all_labels, cm