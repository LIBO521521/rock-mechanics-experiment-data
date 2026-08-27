import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow.keras.layers
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense, Conv2DTranspose, UpSampling2D, concatenate, Input, GlobalAveragePooling2D, Reshape, Multiply
from tensorflow.keras import Model
from matplotlib import rcParams


# ====================== 数据集加载 ======================
x_train_path = './train_processeds/'
y_train_path = './train_labels/'
train_txt = './train and label.txt'


def generateds(path, label, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y_ = [], []
    for content in contents:
        value = content.split()
        data_path = path + value[0]
        label_path = label + value[1]

        with open(data_path, "r") as f:
            data = f.read().split("\n")
        del data[-1]
        data = [float(i) for i in data]
        mean = np.mean(data)
        data = [d - mean for d in data]
        minv, maxv = min(data), max(data)
        if maxv != minv:
            data = [(d - minv) / (maxv - minv) for d in data]
        a = np.array(data).reshape(-1, 1)
        x.append(a)

        with open(label_path, "r") as f:
            data1 = f.read().split("\n")
        del data1[-1]
        b = np.array([float(i) for i in data1]).reshape(-1, 1)
        y_.append(b)
        print('loading : ' + content)
    return np.array(x), np.array(y_)


x_data, y_data = generateds(x_train_path, y_train_path, train_txt)
np.random.seed(100)
np.random.shuffle(x_data)
np.random.seed(100)
np.random.shuffle(y_data)
tf.random.set_seed(100)

x_train = x_data[:-1555]
y_train = y_data[:-1555]
x_test = x_data[-1555:]
y_test = y_data[-1555:]

x_train = x_train[..., tf.newaxis]
y_train = y_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_test = y_test[..., tf.newaxis]

print("x_train shape:", x_train.shape)


# ====================== SE 注意力机制 ======================
def se_block(input_feature, ratio=8):
    """
    Squeeze-and-Excitation Block
    """
    channel = input_feature.shape[-1]
    se = GlobalAveragePooling2D()(input_feature)
    se = Reshape((1, 1, channel))(se)
    se = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([input_feature, se])
    return x


# ====================== U-Net + SE ======================
def unet(pretrained_weights=None, input_size=(1024, 1, 1)):
    inputs = Input(input_size)

    # ---------- Encoder ----------
    conv1 = Conv2D(64, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = se_block(conv1)
    pool1 = MaxPooling2D(pool_size=(2,1))(conv1)

    conv2 = Conv2D(128, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = se_block(conv2)
    pool2 = MaxPooling2D(pool_size=(2,1))(conv2)

    conv3 = Conv2D(256, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = se_block(conv3)
    pool3 = MaxPooling2D(pool_size=(2,1))(conv3)

    conv4 = Conv2D(512, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = se_block(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,1))(drop4)

    conv5 = Conv2D(1024, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = se_block(conv5)
    drop5 = Dropout(0.5)(conv5)

    # ---------- Decoder ----------
    up6 = Conv2D(512, (2,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,1))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = se_block(conv6)

    up7 = Conv2D(256, (2,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,1))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = se_block(conv7)

    up8 = Conv2D(128, (2,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,1))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = se_block(conv8)

    up9 = Conv2D(64, (2,1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,1))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = se_block(conv9)

    conv9 = Conv2D(2, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1,1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


# ====================== 训练 ======================
model = unet()

checkpoint_save_path = "./checkpoint/MSCNN_SE.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=60,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

# ====================== 可视化 ======================
history_dict = history.history
loss_value = history_dict["loss"]
val_loss_value = history_dict["val_loss"]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss_value)+1)

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs, loss_value, "r", label="Training loss")
plt.plot(epochs, val_loss_value, "b", label="Validation loss")
plt.legend()
plt.show()

plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, acc, "r", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.legend()
plt.show()
