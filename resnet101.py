import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
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
tmi_files = glob.glob('C:/Users/anton/Downloads/augmented/tmi/*.png')
wsi_files = glob.glob('C:/Users/anton/Downloads/augmented/wsi/*.png')

# Get the class labels
tmi_labels = [get_class(f) for f in tmi_files]
wsi_labels = [get_class(f) for f in wsi_files]

# Split the datasets into training and validation sets
tmi_train_files, tmi_val_files, tmi_train_labels, tmi_val_labels = train_test_split(tmi_files, tmi_labels, test_size=0.1)
wsi_train_files, wsi_val_files, wsi_train_labels, wsi_val_labels = train_test_split(wsi_files, wsi_labels, test_size=0.1)

# Combine the TMI and WSI datasets
train_files = tmi_train_files + wsi_train_files
val_files = tmi_val_files + wsi_val_files
train_labels = tmi_train_labels + wsi_train_labels
val_labels = tmi_val_labels + wsi_val_labels

# Create an image data generator
datagen = ImageDataGenerator(rescale=1./255)

# Create the training and validation data generators
train_gen = datagen.flow_from_directory(directory="/gpfs/space/home/amlk/data/augmented/", target_size=(512, 512), batch_size=32, class_mode='categorical')
val_gen = datagen.flow_from_directory(directory='/gpfs/space/home/amlk/data/augmented/', target_size=(512, 512), batch_size=32, class_mode='categorical')

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