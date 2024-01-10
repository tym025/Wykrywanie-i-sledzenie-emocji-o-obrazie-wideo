import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from sklearn.model_selection import train_test_split
from keras import backend as K
from matplotlib import pyplot as plt
import cv2
import dlib
import random
import datetime
import tensorflow as tf

devices = tf.config.experimental.list_physical_devices()
print("Dostępne urządzenia:", devices)
chosen_device = "/physical_device:GPU:0" 
tf.config.experimental.set_visible_devices([], 'GPU')

base_folder = "img_align_celeba/"
input_shape = (95,95,1)

identity = pd.read_csv(base_folder+'sorted_file.txt', sep=" ", header=None, names=['image', 'id'])
partition = pd.read_csv(base_folder+'partition.txt', sep=" ", header=None, names=['image', 'split'])

# Połącz dataframy
data = pd.merge(identity, partition, on='image')

data_sample = data.head(len(data))

# Split danych na train/valid/test
train_data, valid_data = train_test_split(data_sample, test_size=0.2, random_state=42)
test_data = train_test_split(valid_data, test_size=0.5, random_state=42)

images_limit = len(train_data)

# Próbka danych
def show_sample():
    sample_generator = train_generator.generate_batch()
    sample_batch, sample_labels = next(sample_generator)
    sample_images_1, sample_images_2 = sample_batch

    num_samples_to_plot = 4
    for i in range(num_samples_to_plot):
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(sample_images_1[i])
        plt.title(f"Image 1, Label: {sample_labels[i]}")

        plt.subplot(1, 2, 2)
        plt.imshow(sample_images_2[i])
        plt.title(f"Image 2, Label: {sample_labels[i]}")

        plt.show()

# Tworzenie architektury sieci
def create_siamese_network13(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))    
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256, activation='sigmoid'))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1,activation='sigmoid')(L1_distance)

    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    return siamese_net

class SiameseDataGenerator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.indexes = np.arange(images_limit)
        self.pairs, self.pair_labels = self.create_pairs()
        self.mtcnn_run = 0
        self.detector = dlib.get_frontal_face_detector()

    def create_pairs(self):
        pairs = []
        labels = []
        images_limit_local = images_limit
        if(len(self.data) < images_limit): images_limit_local = len(self.data) - 1
        for i in range(images_limit_local):
            anchor_image_path = base_folder + 'img_align_celeba/' + self.data.iloc[i]['image']
            current_label = self.data.iloc[i]['id']
            positive_indices = np.where(self.data['id'] == current_label)[0]
            negative_indices = np.where(self.data['id'] != current_label)[0]

            if len(positive_indices) > 1 and len(negative_indices) > 0:
                positive_indices = positive_indices[positive_indices != i]
                positive_index = np.random.choice(positive_indices)
                positive_image_path = base_folder + 'img_align_celeba/' + self.data.iloc[positive_index]['image']

                negative_index = np.random.choice(negative_indices)
                negative_image_path = base_folder + 'img_align_celeba/' + self.data.iloc[negative_index]['image']

                pairs.append([anchor_image_path, positive_image_path])
                labels.append(1)
                pairs.append([anchor_image_path, negative_image_path])
                labels.append(0)

        return np.array(pairs), np.array(labels)

    # Przetwarzanie wstępne
    def preprocess_image(self, image_path):
        img = load_img(image_path)
        img = np.array(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)

        biggest = 0
        if faces:
            for face in faces:
                bbox = [face.left(), face.top(), face.width(), face.height()]

                # Powiększ o 20% 
                bbox[0] = max(0, int(bbox[0] - 0.1 * bbox[2]))
                bbox[1] = max(0, int(bbox[1] - 0.1 * bbox[3]))
                bbox[2] = min(img.shape[1] - bbox[0], int(1.1 * bbox[2]))
                bbox[3] = min(img.shape[0] - bbox[1], int(1.1 * bbox[3]))

                area = bbox[3] * bbox[2]

                if area > biggest:
                    biggest = area
                    biggest_bbox = bbox

            img = img[biggest_bbox[1]:biggest_bbox[1]+biggest_bbox[3], biggest_bbox[0]:biggest_bbox[0]+biggest_bbox[2]]

        else:
            height, width = img.shape[:2]

            # Manualny crop
            new_height = int(height / 1.4)
            new_width = int(width / 1.4)
            crop_top = (height - new_height) // 2
            crop_bottom = height - crop_top
            crop_left = (width - new_width) // 2
            crop_right = width - crop_left

            img = img[crop_top:crop_bottom, crop_left:crop_right]

        img = img_to_array(img)
        img = cv2.resize(img, (95, 95))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype('float32') / 255.0
        return img

    def generate_batch(self):
        while True:
            np.random.shuffle(self.indexes)
            for i in range(0, images_limit, self.batch_size):
                batch_indexes = self.indexes[i:i + self.batch_size]
                batch_pairs = self.pairs[batch_indexes]
                batch_labels = self.pair_labels[batch_indexes]

                batch_images_1 = [self.preprocess_image(image_path) for image_path in batch_pairs[:, 0]]
                batch_images_2 = [self.preprocess_image(image_path) for image_path in batch_pairs[:, 1]]

                yield [np.array(batch_images_1), np.array(batch_images_2)], batch_labels

batch_size = 16
steps_in_epoch = 500

# Tworzenie sieci i paramtrów treningu
decay_steps = 150

lr_schedule_piece = schedules.PiecewiseConstantDecay(
    [10*500, 20*500, 30*500], [1.5e-4,1e-4,0.75e-4, 5e-5]
)

siamese_models = [create_siamese_network13(input_shape)]

early_stopping = EarlyStopping(monitor='val_loss', patience=65, restore_best_weights=True)

# Trening modelu
for i, siamese_model in enumerate(siamese_models):
    try:
        print(f"\nTraining Siamese Model {i+1}")
        
        date_today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'training_{date_today}'+'/siamese_logs'

        checkpoint_saving = ModelCheckpoint(filepath=f'training_{date_today}'+'/siamese_epoch_{epoch:02d}.h5', save_freq=4)
        logs_saving = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        siamese_model.compile(optimizer=Adam(lr_schedule_piece), loss="binary_crossentropy", metrics="accuracy")

        train_generator = SiameseDataGenerator(train_data, batch_size)
        test_generator = SiameseDataGenerator(valid_data, batch_size)

        train_steps_per_epoch = len(train_data) // batch_size
        test_steps_per_epoch = len(valid_data) // batch_size

        siamese_model.summary()

        siamese_model.fit(train_generator.generate_batch(), steps_per_epoch=steps_in_epoch, epochs=70,
                        validation_data=test_generator.generate_batch(), validation_steps=50,
                        callbacks=[early_stopping, checkpoint_saving, logs_saving])
        
        siamese_model.save(f"training_{date_today}/model{i+1}.h5")

        sequential_layer = siamese_model.get_layer("sequential")
        vectorizer_model = Model(inputs=siamese_model.input[0], outputs=sequential_layer.get_output_at(0))

        vectorizer_model.save(f"training_{date_today}/face_recon_model{i+1}.h5")
    except:
        print(f"Błąd. Model {i+1}")
        continue
