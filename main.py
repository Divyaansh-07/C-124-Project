import cv2
import numpy as np
import tensorflow as tf
mymodel = tf.keras.models.load_model('keras_model.h5')
print(mymodel)
camera = cv2.VideoCapture(0)
while True:
    status, frame = camera.read()
    if status:
        frame = cv2.flip(frame, 1)
        myimage = cv2.resize(frame, (224, 224))
        test_image = np.array(myimage, dtype=np.float32)
        print(test_image)
        expand_image = np.expand_dims(test_image, axis=0)
        print(expand_image)
        normalize_image = expand_image / 255.0
        print(normalize_image)
        prediction = mymodel.predict(normalize_image)
        print("Prediction:", prediction)
        cv2.imshow("Feed", frame)
        code = cv2.waitKey(1)
        if code == 32:
            break
camera.release()
cv2.destroyAllWindows()