from torch import nn

model = nn.Sequential(
    nn.Conv3d(n_class, 32), 
    nn.BatchNorm3d()
    nn.ReLU(),

    nn.Conv3d(32, 64), 
    nn.BatchNorm3d()
    nn.ReLU(),
    #skip connection 1
    nn.MaxPool3d(),

    nn.Conv3d(64, 64),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.Conv3d(64, 128),
    nn.BatchNorm3d(),
    nn.ReLU(),
    #skip connection 2
    nn.MaxPool3d(),

    nn.Conv3d(128, 128),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.Conv3d(128, 256),
    nn.BatchNorm3d(),
    nn.ReLU(),  
    #skip connection 3
    nn.MaxPool3d(),  

    nn.Conv3d(256, 256),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.Conv3d(256, 512),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.ConvTranspose3d(512, 512)
    #concat with skip 3

    nn.Conv3d(768, 256),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.Conv3d(256, 256),
    nn.BatchNorm3d(),
    nn.ReLU(),
    nn.ConvTranspose3d(256, 256)
    #concat with skip 2

    nn.Conv3d(384, 128),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.Conv3d(128, 128),
    nn.BatchNorm3d(),
    nn.ReLU(),
    nn.ConvTranspose3d(128, 128)
    #concat with skip 1

    nn.Conv3d(172, 64),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.Conv3d(64, 64),
    nn.BatchNorm3d(),
    nn.ReLU(),

    nn.Conv3d(64, n_class)


    # #fastai basic block
    # nn.Conv3d(),
    # nn.BatchNorm3d(),
    # nn.ReLU(),
    # nn.Conv3d(), 
    # nn.BatchNorm3d(),

    # Mergelayer(),



    )


build_model(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_model(input_layer, 16)