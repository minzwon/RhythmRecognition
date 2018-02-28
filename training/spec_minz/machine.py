from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Dense, Flatten, GlobalAveragePooling2D, concatenate
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam

class Network:
    def __init__(self, with_compile=True, num_tags=9):
        self.num_tags = num_tags
        self.regularizer = l2(1e-5)
        self.model = self.get_model()


    def get_model(self, with_compile=True):
        # input layer
        audio_input = Input(shape=(96, 512, 1), name='input')
        x = audio_input

        for i in range(5):
            x = Conv2D(32, (3,3), padding='same', kernel_regularizer=self.regularizer,
                       name = 'conv_%d' % i)(x)
            x = BatchNormalization(axis=3, name='BN_%d' % i)(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_%d'%i)(x)
            x = Dropout(0.2)(x)
        x = Conv2D(32, (3,3), padding='same', kernel_regularizer=self.regularizer,
                    name = 'conv_%d' % 6)(x)
        x = BatchNormalization(axis=3, name='BN_%d' % 6)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3,2), strides=(3,2), name='MP_%d'%6)(x)
        x = Dropout(0.2)(x)
        for i in range(7, 10):
            x = Conv2D(32, (3,3), padding='same', kernel_regularizer=self.regularizer,
                       name = 'conv_%d' %i)(x)
            x = BatchNormalization(axis=3, name='BN_%d' % i)(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(1,2), strides=(1,2), name='MP_%d'%i)(x)
            x = Dropout(0.2)(x)

        x = Flatten()(x)


        x = Dense(self.num_tags, kernel_initializer='he_uniform', kernel_regularizer=self.regularizer, name='prediction')(x)
        output = Activation('softmax')(x)

        model = Model(audio_input, output)

        if with_compile:
            optimizer = Adam()
            model.compile(optimizer=optimizer,
                          loss = 'categorical_crossentropy',
                          metrics=['accuracy'])
        return model
