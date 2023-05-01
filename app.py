from flask import Flask, render_template, request
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load the trained model
pca = joblib.load('pca.joblib')
model = joblib.load('brain_joblib.joblib')



# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the HTML form
    file = request.files['image']

    # Save the image to the server
    img_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename
    file.save(img_path)
    imgname = file.filename

    # Preprocess the image
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (200, 200))

    x = img.reshape(1, -1)
    x = x / 255
    x_pca = pca.transform(x)
    prediction = model.predict(x_pca)
    #Thresholding
    img = cv2.imread(img_path, 0)

# 1-level thresholding
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('static/thresh/thresh1.jpg', thresh1)

    # 2-level thresholding
    ret1, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(thresh2, 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('static/thresh/thresh2.jpg', thresh2)

    # 3-level thresholding
    ret1, thresh3 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    ret2, thresh3 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    ret3, thresh3 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('static/thresh/thresh3.jpg', thresh3)

    #CONTOUR 
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image
    contour_img = np.zeros_like(img)


    # Draw the contours on the blank image
    contra = cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

    cv2.imwrite('static/contour/contra.jpg', contra)

    #EDGE DETECTION

    img = cv2.imread(img_path, 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('static/edges/edges.jpg', edges)



    #HISTOGRAM EQUALIZER

    img = cv2.imread(img_path, 0)

# Apply histogram equalization
    equalized_img = cv2.equalizeHist(img)
    cv2.imwrite('static/histogram/histogram.jpg', equalized_img)
# Map the predicted label to the class name
    classes = {0: 'no_tumor', 1: 'pituitary_tumor', 2: 'meningioma_tumor', 3: 'glioma_tumor'}
    predicted_class = classes[prediction[0]]
    


    # Return the predicted class
    return render_template('index.html', prediction_text='The predicted class is {}'.format(predicted_class),imgname=imgname)




if __name__ == '__main__':
    app.run(debug=True)