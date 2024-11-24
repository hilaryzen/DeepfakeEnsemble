import numpy as np
from classifiers import *
from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('MesoNet/weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'ondemand',
        target_size=(256, 256),
        batch_size=200,
        class_mode='binary',
        subset='training')

# 3 - Predict
# X, y = next(generator)
for i in range(2):
  X, y = next(generator)
  # print('Predicted :', classifier.predict(X), '\nReal class :', y)
  print(classifier.get_accuracy(X, y))

#scores = classifier.evaluate(generator)
#print("%s%s: %.2f%%" % ("evaluate ",classifier.metrics_names[1], scores[1]*100))

# 4 - Prediction for a video dataset

# classifier.load('weights/Meso4_F2F.h5')

# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])