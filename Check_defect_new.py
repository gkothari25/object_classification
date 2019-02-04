from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications import inception_resnet_v2
from keras.preprocessing.image import ImageDataGenerator
import os ,sys

x_train = []
# Load the training data set by looping over every image file
for image_file in Path("training_dataset").glob("**/*.jpg"):
    #print(image_file)
    
    
    # Load the current image file
    image_data = image.load_img(image_file, target_size=(73, 73))

    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)

    # Add the current image to our list of training images
    x_train.append(image_array)


print(len(x_train))

# Convert the list of separate images into a single 4D numpy array. This is what Keras expects.
x_train = np.array(x_train)

# Normalize image data to 0-to-1 range
x_train = xception.preprocess_input(x_train)

# Add a label for this image. If it was a Defect, label it 0. If it was a Healthy, label it 1.
y_train = []
path = "training_dataset/"
classes = ["Defect","Healthy"]
for i in classes:
    if i == "Defect":
        new_path = os.path.join(path,i)
        y = os.listdir(new_path)
        for i in y:
            y_train.append(0)
    else:
        new_path = os.path.join(path,i)
        y = os.listdir(new_path)
        for i in y:
            y_train.append(1)
print(len(y_train))

# Load the pre-trained neural network to use as a feature extractor
feature_extractor = xception.Xception(weights='imagenet', include_top=False, input_shape=(73, 73, 3))
#x = feature_extractor

#feature_extractor = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))


#feature_extractor = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(75, 75, 3))


# Extract features for each image (all in one pass)
features_x = feature_extractor.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "model/x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "model/y_train.dat")

# Load data set of extracted features
x_train = joblib.load("model/x_train.dat")
y_train = joblib.load("model/y_train.dat")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# Create a model and add layers
model = Sequential()

# Add layers to our model
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    validation_split=0.05,
    epochs=13,
    shuffle=True,
    verbose=2
)

# Save the trained model to a file so we can use it to make predictions later
model.save("model/Defect_healthy_classifier_model.h5")


from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

#Add a label for this image. If it was a Defect, label it 0. If it was a Healthy, label it 1.
y_test = []
path = "test_dataset/"
classes = ["Defect","Healthy"]
for i in classes:
    if i == "Defect":
        new_path = os.path.join(path,i)
        y = os.listdir(new_path)
        for i in y:
            y_test.append(0)
    else:
        new_path = os.path.join(path,i)
        y = os.listdir(new_path)
        for i in y:
            y_test.append(1)


# Empty lists to hold the images and labels for each each image
x_test = []

# Load the test data set by looping over every image file
for image_file in Path("test_dataset").glob("**/*.jpg"):

    # Load the current image file
    image_data = image.load_img(image_file, target_size=(73, 73))

    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)

    # Add the current image to our list of test images
    x_test.append(image_array)
    
# Convert the list of test images to a numpy array
x_test = np.array(x_test)

# Normalize test data set to 0-to-1 range
x_test = xception.preprocess_input(x_test)

print(x_test.shape)

# Load our trained classifier model
model = load_model("model/Defect_healthy_classifier_model.h5")
# Extract features for each image (all in one pass)
features_x = feature_extractor.predict(x_test)

print(features_x.shape)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# Given the extracted features, make a final prediction using our own model
predictions = model.predict(features_x)
#print(predictions)

# If the model is more than 50% sure the object is a Defect, call it a Defect.
# Otherwise, call it "Healthy".
predictions = predictions > 0.5

# Calculate how many mis-classifications the model makes
tn, fp, fn, tp = confusion_matrix(y_test, predictions,labels=[0,1]).ravel()
print("True negative",tn)
print("False positive",fp)
print("False negative",fn)
print("True Positive",tp)

accuracy_score = accuracy_score(y_test, predictions)
print("Accuracy score percent is :",accuracy_score)

f1_score = f1_score(y_test, predictions)
print("f1_score percent is :",f1_score)

precision_score = precision_score(y_test, predictions)
print("precision_score percent is :",precision_score)

recall_score = recall_score(y_test, predictions)
print("recall_score percent is :",recall_score)

 





