import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculateAccuracy(outputs, predicted):
    total = outputs.size(0)
    correct += (predicted == labels).sum().item()
    return (100 * correct / total)
    
def evaluate(model, testloader):
    correct=0

    model.eval()
    outputs = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'].to(device), data['labels'].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
    accuracy = calculateAccuracy(outputs, predicted)
    print('Accuracy on test images: %d %%' % accuracy)
