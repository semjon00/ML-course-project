import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
import glob

# Define the class names
class_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC', 'OTHER']

# Define a function to get the class label from the filename
def get_class(file_path):
    txt_file = file_path.replace('.png', '.txt')
    with open(txt_file, 'r') as f:
        class_label = f.read().strip().split()[0]  # split the line into parts and take the first part
    return class_label

# Get the list of image files
tmi_files = glob.glob('/gpfs/space/home/amlk/data/augmented/tmi/*.png')


# Get the class labels
tmi_labels = [get_class(f) for f in tmi_files]


# Combine the TMI and WSI datasets
all_files = tmi_files 
all_labels = tmi_labels

# Convert class labels to integer indices
encoder = LabelEncoder()
all_labels_encoded = encoder.fit_transform(all_labels)

# Split the datasets into training and validation sets
train_files, val_files, train_labels_encoded, val_labels_encoded = train_test_split(all_files, all_labels_encoded, test_size=0.1)

# Convert integer indices to one-hot encoded labels
train_labels_onehot = to_categorical(train_labels_encoded, num_classes=6)
val_labels_onehot = to_categorical(val_labels_encoded, num_classes=6)

# Create an image data generator
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_files, labels, batch_size, dim, n_classes):
        self.image_files = image_files
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        # Calculate the start and end indices for this batch
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_files))

        # Get the list of image files for this batch
        image_files_temp = self.image_files[start_idx:end_idx]

        # Get the corresponding labels
        labels_temp = self.labels[start_idx:end_idx]

        # Generate the data for this batch
        X, y = self.__data_generation(image_files_temp, labels_temp)

        return X, y

    def __data_generation(self, image_files_temp, labels_temp):
        X = np.empty((len(image_files_temp), *self.dim, 3))
        y = np.empty((len(labels_temp), self.n_classes))

        for i, image_file in enumerate(image_files_temp):
            img = tf.keras.preprocessing.image.load_img(image_file, target_size=self.dim)
            X[i,] = tf.keras.preprocessing.image.img_to_array(img)
            y[i] = labels_temp[i]

        return X / 255.0, y

# Create the training and validation data generators
train_gen = CustomDataGenerator(train_files, train_labels_onehot, batch_size=32, dim=(512, 512), n_classes=6)
val_gen = CustomDataGenerator(val_files, val_labels_onehot, batch_size=32, dim=(512, 512), n_classes=6)

# Load the pretrained ResNet-101 model
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Make the base model trainable
base_model.trainable = True

# Add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = tf.keras.layers.Dense(1024, activation='relu')(x)

# Add a logistic layer with 6 classes (we have 6 different classes)
predictions = tf.keras.layers.Dense(6, activation='softmax')(x)

# This is the model we will train
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=1)

weights_dir = '/gpfs/space/home/amlk/data'

# Make sure the directory exists
os.makedirs(weights_dir, exist_ok=True)

# Save the weights
model.save_weights(os.path.join(weights_dir, 'my_model_weights.h5'))