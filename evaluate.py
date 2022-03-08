import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculateAccuracy(labels, pred_labels):
    total = labels.size(0)
    correct += (predicted == labels).sum().item()
    return (100 * correct / total)

def calculateConfusion(labels, pred_labels):
    for i in range(labels.size(0)):
        divide = pred_labels/labels

        true_positives = torch.sum(divide == 1).item()
        false_positives = torch.sum(divide == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(divide)).item()
        false_negatives = torch.sum(divide == 0).item()

        return true_positives, false_positives, true_negatives, false_negatives

        
def calculatePrecision(labels, pred_labels):
    true_positives, false_positives, true_negatives, false_negatives = calculateConfusion(labels, pred_labels)
    return true_positives/(true_positives + false_positives)
    #precision Tp/(Tp + Fp)

def calculateRecall(labels, pred_labels):
    true_positives, false_positives, true_negatives, false_negatives = calculateConfusion(labels, pred_labels)
    return true_positives/(true_positives + false_negatives)
    #precision Tp/(Tp + Fn) ALSO known as sensitivity

def calculateF1(labels, pred_labels):
    precision = calculatePrecision(labels, pred_labels)
    recall = calculateRecall(labels, pred_labels)
    return precision*recall/(precision+recall)

def calculateSpecificity(labels, pred_labels):
    true_positives, false_positives, true_negatives, false_negatives = calculateConfusion(labels, pred_labels)
    return true_negatives/(true_negatives + false_positives)
    
def evaluate(model, testloader):

    pred_labels = []
    actual_labels = []

    model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'].to(device), data['labels'].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_list = predicted.tolist()
            pred_labels +=predicted_list
            labels_list = labels.tolist()
            labels += labels_list
            
    accuracy = calculateAccuracy(actual_labels, labels)
    precision = calculatePrecision(labels, pred_labels)
    recall = calculateRecall(labels, pred_labels)
    F1 = calculateF1(labels, pred_labels)
    specificity = calculateSpecificity(labels, pred_labels)
    
    print('Accuracy on test images: %d %%' % accuracy)
    print('Precision on test images: %d %%' % precision)
    print('Recall on test images: %d %%' % recall)
    print('F1 on test images: %d %%' % F1)
    print('Specificty on test images: %d %%' % specificity)

    return accuracy, precision, recall, F1, specificity 



    
