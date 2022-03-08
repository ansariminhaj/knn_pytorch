import torch.nn as nn
import torch.nn.functional as F

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

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=1e-4)

def build_unet(config): # not a unet this is our modified 12 lead weston network
    import keras
    from keras.layers import Conv1D, BatchNormalization, Add, MaxPooling1D
    from keras.optimizers import Adam
    from keras.layers.core import Dense, Activation, Flatten
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dropout
    
    
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    
    r = lambda: keras.regularizers.l2(config["l2_kern_weight"])

    layer = nn.Conv1d(
             filters=config["conv_num_filters_start"],
             kernel_size=config["conv_filter_length"],
             stride=1,
             padding='same', 
             kernel_regularizer=r(), 
            # weston does not use kernel initalizer 
             kernel_initializer=params["conv_init"])(inputs)
    
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    layer2 = Conv1D(
             filters=params["conv_num_filters_start"],
             kernel_size=params["conv_filter_length"],
             strides=1,
             padding='same', 
             kernel_regularizer=r(), 
            # weston does not use kernel initalizer 
             kernel_initializer=params["conv_init"])(layer)

    layer2 = BatchNormalization()(layer2)
    layer2 = Activation(params["conv_activation"])(layer2)
    layer2 = Dropout(params["conv_dropout"])(layer2)
    layer2 = Conv1D(
                 filters=params["conv_num_filters_start"],
                 kernel_size=params["conv_filter_length"],
                 strides=1,
                 padding='same', 
                 kernel_regularizer=r(), 
                # weston does not use kernel initalizer 
                 kernel_initializer=params["conv_init"])(layer2)
    
    layer = Add()([layer,layer2])
    # x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='same')
    layer = MaxPooling1D(pool_size=2, strides=2,padding='same')(layer)
    
    for i in range(1, 1 + params["num_middle_layers"]):
        layer2 = layer
        n_filters = params["conv_num_filters_start"] * 2 ** (i // params["conv_increase_channels_at"])

        for j in range(params["num_convs_per_layer"]):
            layer2 = BatchNormalization()(layer2)
            layer2 = Activation(params["conv_activation"])(layer2)
            layer2 = Dropout(params["conv_dropout"])(layer2)
            layer2 = Conv1D(
                 filters=n_filters,
                 kernel_size=params["conv_filter_length"],
                 strides=1,
                 padding='same', 
                 kernel_regularizer=r(), 
                 kernel_initializer=params["conv_init"])(layer2)
            
        if i % params["conv_increase_channels_at"] == 0:
            layer = layer2
        else:
            layer = Add()([layer,layer2])

        if i % params["conv_pool_at"] == 0:
            # add padding = 'same'
            layer = MaxPooling1D(pool_size=2,strides=2,padding='same')(layer)

    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)
    layer = Flatten()(layer)
    for i in range(params["hidden_layers"]):
        layer = Dense(params["hidden_size"])(layer)

    layer = Dense(params["num_categories"])(layer)
    
    output = Activation("sigmoid")(layer)
    
    model = Model(inputs=[inputs], outputs=[output])

    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc', f1_score])
    
    return model
