import torch.nn as nn
import torch.nn.functional as F
from load_dataset import create_dataloaders

# "input_shape":[2048,12]
# This means 12 ECG nodes, 2048 time steps

# Conv, Batchnorm and Activation
def layerA(in_channels, out_channels, activation, *args, **kwargs):
    activation_function = getattr(nn, activation)() 
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm1d(out_f),
        activation_function
    )

# Conv, Batchnorm, Activation and Dropout
def layerB(in_channels, out_channels, activation, dropout, *args, **kwargs):
    activation_function = getattr(nn, activation)() 
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm1d(out_f),
        activation_function,
        nn.Dropout(p=dropout)
    )

class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        def encoderFunction():

            layer = [layerA(config['input_shape'][0], config['conv_num_filters_start'], config['conv_activation'],
                                         kernel_size=config['conv_filter_length'], padding='same', stride=1),
                      layerB(config['conv_num_filters_start'], config['conv_num_filters_start'], config['conv_activation'],
                             config['conv_dropout'], kernel_size=config['conv_filter_length'], padding='same', stride=1),
                      layerB(config['conv_num_filters_start'], config['conv_num_filters_start'], config['conv_activation'],
                             config['conv_dropout'], kernel_size=config['conv_filter_length'], padding='same', stride=1)]
            
            for i in range(1, 1 + params["num_middle_layers"]):
                layer2 = layer

                filter_multiple = 2 ** (i // config["conv_increase_channels_at"])
                n_filters = config["conv_num_filters_start"] * filter_multiple
                
                for j in range(config["num_convs_per_layer"]):
                    layer2.append(layerB( n_filters / filter_multiple, n_filters, kernel_size=config['conv_filter_length'], padding='same'))

                if i % config["conv_increase_channels_at"] == 0:
                    layer = layer2
                else:
                    layer = [layer, layer2]

                if i % config["conv_pool_at"] == 0:
                    layer.append(nn.MaxPool1d(2, stride=2, paddng='same'))
                    
            return layer

        def decoderFunction():

            layer = []
            layer.append(nn.Flatten(start_dim = 0, end_dim = -1))

            for i in range(config["hidden_layers"]):
                layer.append(nn.Linear(x.shape[0], config["hidden_size"]))

            layer.append(nn.Linear(x.shape[0], config["num_categories"]))
            layer.append(nn.Sigmoid())

            return layer
        
        self.encoder = nn.Sequential(*encoderFunction())
        self.decoder = nn.Sequential(*decoderFunction())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(train_dataloader, val_dataloader, config):
    net = Net()
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

    # get the inputs; data is a dict {'image', 'labels'}
    for epoch in range(config['epoch']):  
        count_t=0
        count_v=0
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            net.train()

            inputs, labels = data['image'].to(device), data['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count_t+=1
            
        train_loss.append(running_loss/count_t)
        print("Epoch ", epoch, "Train loss: ",running_loss/count_t)
                
        running_loss_v = 0.0     
        for i_v, data_v in enumerate(testloader, 0):
            net.eval()
            inputs_v, labels_v = data_v[0].to(device), data_v[1].to(device)

            outputs_v = net(inputs_v)
            loss_v = criterion(outputs_v, labels_v)

            running_loss_v += loss_v.item()
            count_v += 1 
            
        
        val_loss.append(running_loss_v/count_v)
        print("Epoch ", epoch, "Val loss: ",running_loss_v/count_v)
            
        if len(val_loss) == 1:
            print("Initial val loss saved\n")
            torch.save(net.state_dict(), "weights/")
        else:
            if val_loss[-1] <= min(val_loss):
                print("Val loss saved\n")
                torch.save(net.state_dict(), "weights/"+str(epoch)+"_.pth")
            else:
                print("\n")

    print('Finished Training')
    return net
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    
    config = read_config(sys.argv[1])

    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(config)
    
    model = train(train_dataloader, val_dataloader, config)
    


