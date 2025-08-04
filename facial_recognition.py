import deepface.DeepFace
import depthai as dai
import cv2
import numpy as np
import deepface
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from deepface import DeepFace
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf

training_dataset = {} #Empty Training Dataset
validation_dataset = {} #Empty Validation Dataset
testing_dataset = {} #Empty Testing Dataset
IMAGE_SIZE = (300, 300) #Default Image Size
BATCH_SIZE = 32 #Default Batch Size
AUTOTUNE = tf.data.AUTOTUNE #Tensorflow constant that sets up the the prefetch buffer size 
#.batch() groups a specified number of dataset elements such as images and labels into a single batch.
#.prefetch() allows the dataset pipeline to prepare the next batch of data while the current batch is getting processed by the model

#Preprocesses the given image
def preprocess(rgb_data, labels):
    rgb_data = tf.cast(rgb_data, tf.float32)/255.0 #Normalizes the images 
    #(Should have used  img = cv2.imread(img_path) img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1] instead) 
    #means.append(np.mean(img, axis=(0, 1)))
    #stds.append(np.std(img, axis=(0, 1)))

# Compute dataset-wide mean and std
    #mean = np.mean(means, axis=0)
    #std = np.mean(stds, axis=0)

    #print("Dataset Mean:", mean)
    #print("Dataset Std:", std)
    labels = tf.one_hot(labels, depth=number_of_classes) #Generates One-Hot Encoding Vectors
    return rgb_data, labels #Returns Normalized RGB Images and Labels

#Aguments the data within the each dataset
data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), #Flips the image horizontally
        tf.keras.layers.RandomRotation(0.2), #Rotates the image randomly by 0.2
        tf.keras.layers.RandomZoom(0.2), #Zooms in on the image randomly by 0.2
])

def augmentation(rgb_data, labels): #Reduces overfitting
    rgb_data = data_augmentation(rgb_data)
    return rgb_data, labels

lr_scheduler = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,verbose=1) #Dynamically reduces the learning rate when validation loss plateaus
early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True) #Stops training when no improvement is observed, saving on resources for computation

training_dataset = tf.keras.utils.image_dataset_from_directory(f"faces/Train/",shuffle=True, image_size=IMAGE_SIZE,batch_size=BATCH_SIZE) #Fully initialzed Training Dataset

class_names = training_dataset.class_names #Generates the class names for the Model
number_of_classes = len(class_names) #Length of class names

training_dataset = training_dataset.map(preprocess).map(augmentation).prefetch(AUTOTUNE) #Preprocesses and augments the data inside the training dataset

validation_dataset =  tf.keras.utils.image_dataset_from_directory(f"faces/Validation/",shuffle=True, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE) #Fully initialzed Validation Dataset
validation_dataset = validation_dataset.map(preprocess).prefetch(AUTOTUNE) #Preprocesses the data inside the validation dataset

testing_dataset =  tf.keras.utils.image_dataset_from_directory(f"faces/Test/",shuffle=False, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE) #Fully initialzed Testing Dataset
testing_dataset = testing_dataset.map(preprocess).prefetch(AUTOTUNE) #Preprocesses the data inside the validation dataset

rgb_input_layer = Input(shape = (300,300,3), name="rgb_input_layer") #Initializes the original input layer within our CNN
rgb_convolutional_layer = Conv2D(filters = 32, kernel_size=(3,3), activation="relu", padding = "same")(rgb_input_layer) #Detects edges, nose, mouth, etc
rgb_convolutional_layer = BatchNormalization()(rgb_convolutional_layer) #Stabilizes training and improves performance
rgb_convolutional_layer = MaxPooling2D(pool_size=(2,2))(rgb_convolutional_layer) #Applies Max Pooling within our current convolutional layer
rgb_convolutional_layer = Conv2D(filters = 64, kernel_size=(3,3), activation="relu", padding = "same")(rgb_convolutional_layer) #Detects arrangment of facial components
rgb_convolutional_layer = BatchNormalization()(rgb_convolutional_layer) #Applies Batch Normalizaton
rgb_convolutional_layer = MaxPooling2D(pool_size=(2,2))(rgb_convolutional_layer) #Applies Max Pooling within our current convolutional layer
rgb_convolutional_layer = Conv2D(filters = 128, kernel_size=(3,3), activation="relu", padding = "same")(rgb_convolutional_layer) #Recognizes the overall structure of the face
rgb_convolutional_layer = BatchNormalization()(rgb_convolutional_layer) #Applies Batch Normalizaton
rgb_convolutional_layer = MaxPooling2D(pool_size=(2,2))(rgb_convolutional_layer) #Applies Max Pooling within our current convolutional layer
rgb_convolutional_layer = Dropout(0.5)(rgb_convolutional_layer) #Adds a dropout layer in our model to reduce overfitting

rgb_convolutional_layer = Flatten()(rgb_convolutional_layer) #Converts these layers into a 1D Vector
rgb_convolutional_layer = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(rgb_convolutional_layer) #Creates a dense (fully connected) layer with 128 neurons.
rgb_convolutional_layer = Dropout(0.5)(rgb_convolutional_layer) #Additional dropout layer added after Dense layer for further regularization
rgb_output_layer = Dense(number_of_classes, activation='softmax')(rgb_convolutional_layer) #Added Dense Layer for proper distribution of our class names. Activated through the given Softmax Function to produce our outputs.

model = Model(rgb_input_layer,rgb_output_layer) #Initialized model with input and output layers passed in.
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"]) #Compiles our model for accuracy, categorizes the images in class names, and optimizes it in adam.
model.summary()

facial_recognition_model = model.fit(training_dataset, validation_data=validation_dataset, epochs = 50, callbacks=[lr_scheduler,early_stopping], verbose=1) #Trains the model with 50 epochs of training and adds a learning rate scheduler/early stopping to prevent overfitting and increase accuracy in our model.
testing_loss, testing_accuracy = model.evaluate(testing_dataset) #Evaluates the model's testing loss and testing accuracy
print(f"Testing Loss: {testing_loss}, Testing Accuracy: {testing_accuracy}")

model_directory = "./facial_recognition_model/" #Directory where our model is stored.
tflite_model_directory = "./facial_recognition_model_tflite" #Directory where our tensorflow lite model is stored.

os.makedirs(model_directory,exist_ok=True) #Verifies the model directory and creates it just in case it is not present.
os.makedirs(tflite_model_directory,exist_ok=True) #Verifies the tensorflow lite model directory and creates it just in case it is not present.

print("Saving TensorFlow Model....")
model.save(model_directory) #Saves the model directory
print(f"Tensorflow Model saved in: {model_directory}")

#Converting Tensorflow Model to tflite Model
print("Converting Tensorflow Model to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(model_directory) #Creates the converter object from the tensorflow model from it's stored directory.
facial_recognition_tflite_model = converter.convert() #Converts the model into a tensorflow light model.

#Writing to tensorflow lite model
tflite_model_path = os.path.join(tflite_model_directory,"facial_recognition_tflite_model.tflite") #Creates the tensorflow lite model directory
with open(tflite_model_path,"wb") as f: #Opens the tensorflow model path and treats it as a regular test file. 
    f.write(facial_recognition_tflite_model) #Writes to the file all of the data present within the original model to the tensorflow lite model.
print(f"TFLite Model saved at: {tflite_model_path}")

'''
early_stopping_model_data = {
    "loss": facial_recognition_model.history["loss"],
    "val_loss": facial_recognition_model.history["val_loss"],
    "accuracy": facial_recognition_model.history["accuracy"],
    "val_accuracy": facial_recognition_model.history["val_accuracy"]
}

# Graph Training Accuracy Vs. Validation Accuracy
plt.figure()
plt.plot(facial_recognition_model.history["accuracy"], label="Training Accuracy")
plt.plot(facial_recognition_model.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy with Early Stopping")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

facial_recognition_model_no_early_stopping_model = model.fit(training_dataset, validation_data=validation_dataset, epochs = 50, verbose=1)
testing_loss_no_early_stopping, testing_accuracy_no_early_loss = model.evaluate(testing_dataset)
print(f"Testing Loss No Early Stopping: {testing_loss_no_early_stopping} , Testing Accuracy No Early Stopping: {testing_accuracy_no_early_loss}")

no_early_stopping_model_data = {
    "loss": facial_recognition_model_no_early_stopping_model.history["loss"],
    "val_loss": facial_recognition_model_no_early_stopping_model.history["val_loss"],
    "accuracy": facial_recognition_model_no_early_stopping_model.history["accuracy"],
    "val_accuracy": facial_recognition_model_no_early_stopping_model.history["val_accuracy"]
}

#Graph Training Accuracy Vs. Validation Accuracy
plt.figure()
plt.plot(early_stopping_model_data["val_loss"], label="Val Loss (Early Stopping)")
plt.plot(no_early_stopping_model_data["val_loss"], label="Val Loss (No Early Stopping)")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#model.fit(x=[rgb_train,depth_train], y=labels, validation_data=([rgb_validation,depth_validation], validation_labels), epochs=20, verbose=32)
'''