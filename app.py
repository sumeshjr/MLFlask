# Python that converts a Python dictionary or other JSON-serializable object into a JSON-formatted response.
from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Load your pre-trained model
json_file = open('models/model_rev.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model_rev.h5")

@app.route('/')
def index():
    return render_template('index.html')

# The model is stored in the loaded_model variable 
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'})
    #cv2.imdecode(...): This is a function from the OpenCV library (cv2) used to decode the image data
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    if img is not None:
        # Invert image
        img = ~img
        # the variable ret is typically used to store a return value from a function
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ctrs, ret = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # a contour refers to a continuous curve or boundary that outlines 
        # or represents the shape of an object or region in an image
        # Sorting on the basis of x coordinate
        # This variable will store the contours of detected objects in the image
        # This line sorts the contours (ctrs) based on their x-coordinate
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        #These lines set the width (w) and height (h) dimensions to 28
        w = int(28)
        h = int(28)
        #train_data: This list will store the resized symbols for later recognition.
        #rects: This list will store the bounding rectangles of the detected symbols.
        train_data = []
        rects = []
        
        #This loop iterates through the detected contours (cnt) 
        # and extracts the bounding rectangle coordinates for each contour.
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            # top-left corner of the bounding rectangle (x and y) and its width and height (w and h).
            rect = [x, y, w, h]
            rects.append(rect)
        
        #For each pair of rectangles (r and rec), it checks if they overlap based on their coordinates. 
        # If they do overlap within a certain margin (10 pixels in this case), 
        # it sets flag to 1; otherwise, it sets flag to 0.
        # 1 for overlap, 0 for no overlap
        bool_rect = []
        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and \
                            rec[1] < (r[1] + r[3] + 10):
                        flag = 1
                    l.append(flag)
                if rec == r:
                    l.append(0)
            bool_rect.append(l)
        
        # Remove the overlapping rectangles
        # These nested loops iterate through all pairs of bounding rectangles. 
        # The outer loop, controlled by i, goes through each bounding rectangle i, 
        # and the inner loop, controlled by j, goes through each bounding rectangle j.
        dump_rect = []
        for i in range(0, len(cnt)):
            for j in range(0, len(cnt)):
                if bool_rect[i][j] == 1:
                    area1 = rects[i][2] * rects[i][3]
                    area2 = rects[j][2] * rects[j][3]
                    if area1 == min(area1, area2):
                        #If the condition is met, it means that rects[i] has the smaller area, 
                        #and therefore, it is considered for removal
                        dump_rect.append(rects[i])
        
        # Final list of rects in which actual digit/symbol is residing
        final_rect = [i for i in rects if i not in dump_rect]

        #This line initializes an empty string to store the recognized mathematical expression
        equation = ''
        
        #x-coordinate, y-coordinate, width, and height of the current bounding rectangle (r).
        for r in final_rect:
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            im_crop = thresh[y:y + h + 10, x:x + w + 10]
            im_resize = cv2.resize(im_crop, (28, 28))

            im_resize = np.reshape(im_resize, (28, 28, 1))
            train_data.append(im_resize)

        for i in range(len(train_data)):
            train_data[i] = np.array(train_data[i])
            train_data[i] = train_data[i].reshape(1, 28, 28, 1)
            result = loaded_model.predict(train_data[i])
            #returns the indices of the maximum values along a specified axis in an array.
            predicted_class = np.argmax(result)
            
            if predicted_class == 10:
                equation += '-'
            elif predicted_class == 11:
                equation += '+'
            elif predicted_class == 13:
                equation += '/'
            elif predicted_class == 12:
                equation += '*'
            elif predicted_class == 0:
                equation += '0'
            elif predicted_class == 1:
                equation += '1'
            elif predicted_class == 2:
                equation += '2'
            elif predicted_class == 3:
                equation += '3'
            elif predicted_class == 4:
                equation += '4'
            elif predicted_class == 5:
                equation += '5'
            elif predicted_class == 6:
                equation += '6'
            elif predicted_class == 7:
                equation += '7'
            elif predicted_class == 8:
                equation += '8'
            elif predicted_class == 9:
                equation += '9'

        try:
            result = eval(equation)
            return render_template('result.html', equation=equation, result=result, error_message=None)
        except Exception as e:
            error_message = "Invalid Equation"
        return render_template('result.html', equation=equation, result=None, error_message=error_message)

    return render_template('result.html', error_message='Unable to process the image')

if __name__ == '__main__':
    app.run(debug=True)
