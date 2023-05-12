import cv2
import tensorflow as tf
import visualkeras
# from numpy import asarray
from PIL import Image

# img = cv2.imread("M:\Mk_Coding\lang_Python\Projects\eve_extras\Test_Images\\snow_cat.jpg")
# img = cv2.imread("M:\Mk_Coding\lang_Python\Projects\eve_extras\Test_Images\\cat_on_fence.jpg")
# img = cv2.imread("M:\Mk_Coding\lang_Python\Projects\eve_extras\Test_Images\\brown_fluffy_dog.jpg")
# res = cv2.resize(img, dsize=(225,225))
#
model = tf.keras.models.load_model("eve_model_2.h5")
#
# numpydata = res / 255
# numpydata = numpydata.reshape(1, 225, 225, 3)
# print("SHAPE = ", numpydata.shape)
#
# prediction = model.predict(numpydata)
# print(prediction)

print(visualkeras.layered_view(model,legend=True).save("model_img.png","PNG"))