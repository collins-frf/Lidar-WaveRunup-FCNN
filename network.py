from lidar_data import *
from settings import *
from losses import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16

class LidarNet(object):
    def __init__(self):
        self.img_rows = dimensions
        self.img_cols = dimensions
        self.features = 4
        self.r2features = 3
        self.val_loss = 500
        self.old_best = 500
        self.fail_counter = 0
        self.test_list = glob.glob('./data/test/*.npy')

    def get_unet(self):

        #make sure to overwrite GaussianNoise layer in source to use in_test_phase not in_train_phase
        inputs = tf.keras.layers.Input((int(.5*self.img_rows), int(.5*self.img_cols), self.features))
        conv1 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
        conv1 = tf.keras.layers.GaussianNoise(0)(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
        conv2 = tf.keras.layers.GaussianNoise(0)(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = tf.keras.layers.GaussianNoise(0)(conv3)
        conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
        conv4 = tf.keras.layers.GaussianNoise(0)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = tf.keras.layers.BatchNormalization(trainable=True)(conv5)
        conv5 = tf.keras.layers.GaussianNoise(0)(conv5)

        up6 = tf.keras.layers.Conv2D(filters, 2, activation=activation, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv5))
        merge6 = tf.keras.layers.concatenate([conv4, up6], axis=3)

        conv6 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = tf.keras.layers.BatchNormalization(trainable=True)(conv6)
        conv6 = tf.keras.layers.GaussianNoise(0)(conv6)

        up7 = tf.keras.layers.Conv2D(filters, 2, activation=activation, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)

        conv7 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
        conv7 = tf.keras.layers.GaussianNoise(0)(conv7)

        up8 = tf.keras.layers.Conv2D(filters, 2, activation=activation, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)

        conv8 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
        conv8 = tf.keras.layers.GaussianNoise(0)(conv8)

        up9 = tf.keras.layers.Conv2D(filters, 2, activation=activation, padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)

        conv9 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = tf.keras.layers.Conv2D(1, 1, activation=None)(conv9)

        model = keras.models.Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=optimizer, loss=loss, metrics=[], run_eagerly=True)

        return model

    def get_VGGnet(self):

        base_model = VGG16(input_shape=(int(.5*self.img_rows), int(.5*self.img_cols), self.r2features),  # Shape of our images
                           include_top=False,  # Leave out the last fully connected layer
                           weights='imagenet')
        # Flatten the output layer to 1 dimension
        x = tf.keras.layers.Flatten()(base_model.output)

        # Add a fully connected layer with 512 hidden units and ReLU activation
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.models.Model(base_model.input, x)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=settings.lr), loss='mean_squared_error', metrics=[])

        return model

    def load_model(self):
        if __name__ == '__main__':
            try:
                iter_saves = glob.glob('./results/*iter.h5')
                iter_saves.sort(key=natural_keys)
                model = self.get_unet()
                model.load_weights(iter_saves[-1])
                print("loaded model ", iter_saves[-1])
            except:
                model = self.get_unet()
                print("couldn't load model")
                model.summary()
                return model, './results/0iter.h5'

            return model, iter_saves[-1]
        else:
            try:
                val_saves = glob.glob('./results/*val_loss.h5')
                val_saves.sort(key=natural_keys)
                model = keras.models.load_model(val_saves[-1])
                print("loaded model ", val_saves[-1])
            except:
                model = self.get_unet()
                model.summary()
                return model, './results/0val_loss.h5'

            return model, val_saves[-1]

    def get_unet_batch(self, dataset, train_flag):
        if train_flag == 'train':
            img_batch = np.ndarray((batch_size, dimensions, dimensions, self.features), dtype=np.float32)
            label_batch = np.ndarray((batch_size, dimensions, dimensions, 1), dtype=np.float32)

            for j in range(batch_size):
                random_index = np.random.uniform(0, len(dataset) - validation_size)
                random_index = int(random_index)
                sample = dataset[random_index]
                sample[()]['label'] = np.expand_dims(sample[()]['label'], axis=-1)
                img_batch[j] = sample[()]['lidar']
                label_batch[j] = sample[()]['label']

        if train_flag == 'validate':
            img_batch = np.ndarray((int(validation_size), dimensions, dimensions, self.features), dtype=np.float32)
            label_batch = np.ndarray((int(validation_size), dimensions, dimensions, 1), dtype=np.float32)
            for j in range(int(validation_size)):
                index = j
                sample = dataset[(len(dataset)-validation_size)+index]
                sample[()]['label'] = np.expand_dims(sample[()]['label'], axis=-1)
                img_batch[j] = sample[()]['lidar']
                label_batch[j] = sample[()]['label']

        if train_flag == 'test':
            img_batch = np.ndarray((int(test_size), dimensions, dimensions, self.features), dtype=np.float32)
            label_batch = np.ndarray((int(test_size), dimensions, dimensions, 1), dtype=np.float32)
            for j in range(int(test_size)):
                index = j
                index += 10000
                sample = dataset[index]
                sample[()]['label'] = np.expand_dims(sample[()]['label'], axis=-1)
                img_batch[j] = sample[()]['lidar']
                label_batch[j] = sample[()]['label']

        return img_batch, label_batch

    def validate(self, dataset, model):

        img_batch, label_batch = self.get_unet_batch(dataset, 'validate')
        final_label_batch = np.zeros((validation_size, int(.5 * dimensions), int(.5 * dimensions)), dtype=np.float32)
        final_img_batch = np.zeros((validation_size, int(.5 * dimensions), int(.5 * dimensions), self.features), dtype=np.float32)

        for l in range(validation_size):
            final_img_batch[l, :, :, 0] = cv2.resize(np.abs(img_batch[l, :, :, 0]/31), (512, 512),
                                                     interpolation=cv2.INTER_AREA)
            final_img_batch[l, :, :, 1] = cv2.resize(img_batch[l, :, :, 1]/7, (512, 512),
                                                     interpolation=cv2.INTER_AREA)
            final_img_batch[l, :, :, 2] = cv2.resize(img_batch[l, :, :, 2], (512, 512), interpolation=cv2.INTER_AREA)
            final_img_batch[l, :, :, 3] = cv2.resize(img_batch[l, :, :, 3], (512, 512), interpolation=cv2.INTER_AREA)
            final_label_batch[l, :, :] = cv2.resize(label_batch[l], (512, 512), interpolation=cv2.INTER_AREA)

        val_history = model.evaluate(final_img_batch, final_label_batch, verbose=0, batch_size=settings.batch_size)
        return val_history

    def log_val(self, val_history, model, epoch):
        writer = tf.summary.create_file_writer(logs_path)
        with writer.as_default():
            tf.summary.scalar("Val_Loss", val_history, step=epoch)
        writer.flush()
        self.val_loss = float(val_history)
        if self.val_loss < self.old_best:
            print(str(self.old_best) + ' was the old best val_loss. ' + str(
                self.val_loss) + ' is the new best val loss!')
            self.old_best = self.val_loss
            model.save('./results/' + str(epoch) + 'val_loss.h5', overwrite=True)
            self.fail_counter = 0
        else:
            self.fail_counter += 1
            print("val better fails in a row: " + str(self.fail_counter))
            if self.fail_counter % 1 == 0:
                print("val loss failed to improve 1 epochs in a row")
                print("Current LR: " + str(model.optimizer.lr) + "reducing learning rate by 1%")
                tf.keras.backend.set_value(model.optimizer.lr, model.optimizer.lr * .99)
                print("New LR: " + str(tf.keras.backend.get_value(model.loss)))

    def train(self):
        dataset = LidarDataset(Dataset)

        model, save = self.load_model()
        #tf.keras.backend.set_learning_phase(1)

        save = save[10:-7]
        writer = tf.summary.create_file_writer(logs_path)
        loss_list = []
        epoch = int(save)
        val_history = self.validate(dataset, model)
        self.old_best = float(val_history)
        print("Loaded Val Loss: ", self.old_best)

        while epoch < epoch_range:
            print("Epoch: ", epoch)
            epoch_loss = []
            if epoch == 5:
                model.compile(optimizer=optimizer, loss=losses.absolute_error_grad, metrics=[], run_eagerly=True)
                self.old_best = 500
            # get training batch
            for batch in range(int(len(dataset)/batch_size)):
                img_batch, label_batch = self.get_unet_batch(dataset, 'train')
                final_label_batch = np.zeros((batch_size, int(.5*dimensions), int(.5*dimensions)), dtype=np.float32)
                final_img_batch = np.zeros((batch_size, int(.5*dimensions), int(.5*dimensions), self.features), dtype=np.float32)
                img_batch = np.where(np.isnan(img_batch), 0, img_batch)
                label_batch = np.where(np.isnan(label_batch), 0, label_batch)

                for l in range(batch_size):
                    final_img_batch[l, :, :, 0] = cv2.resize(np.abs(img_batch[l, :, :, 0]/31), (512, 512), interpolation=cv2.INTER_AREA)
                    final_img_batch[l, :, :, 1] = cv2.resize(img_batch[l, :, :, 1]/7, (512, 512), interpolation=cv2.INTER_AREA)
                    final_img_batch[l, :, :, 2] = cv2.resize(img_batch[l, :, :, 2], (512, 512), interpolation=cv2.INTER_AREA)
                    final_img_batch[l, :, :, 3] = cv2.resize(img_batch[l, :, :, 3], (512, 512), interpolation=cv2.INTER_AREA)
                    final_label_batch[l, :, :] = cv2.resize(label_batch[l], (512, 512), interpolation=cv2.INTER_AREA)
                """fig = plt.figure()
                ax0 = fig.add_subplot(2, 2, 1), plt.imshow(final_img_batch[0, :, :, 0])
                ax0[0].set_ylabel("Cross-shore distance")
                ax0[0].set_xlabel("Time")
                ax1 = fig.add_subplot(2, 2, 2), plt.imshow(final_img_batch[0, :, :, 1])
                ax1[0].set_ylabel("Cross-shore distance")
                ax1[0].set_xlabel("Time")
                ax2 = fig.add_subplot(2, 2, 3), plt.imshow(final_label_batch[0, :, :])
                ax2[0].set_ylabel("Cross-shore distance")
                ax2[0].set_xlabel("Time")
                plt.show()"""
                # train
                train_history = model.train_on_batch(final_img_batch, final_label_batch)
                epoch_loss.append(train_history)
            print("Mean Epoch Loss: ", np.mean(epoch_loss))
            loss_list.append(np.mean(epoch_loss))
            print("Difference: ", np.mean(epoch_loss) - np.mean(loss_list))
            with writer.as_default():
                tf.summary.scalar("Loss", np.mean(epoch_loss), step=epoch)
                tf.summary.scalar("LR", model.optimizer.lr, step=epoch)
            writer.flush()
            # save model at end of each epoch
            if epoch % 1 == 0:
                model.save('./results/' + str(epoch) + 'iter.h5', overwrite=True)
                val_history = self.validate(dataset, model)
                self.log_val(val_history, model, epoch)
            epoch += 1


if __name__ == '__main__':
    net = LidarNet()
    net.train()
