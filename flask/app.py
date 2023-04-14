from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from io import BytesIO
import cv2
import pymongo
import jwt

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

# Connect to the MongoDB Atlas cluster
client = pymongo.MongoClient("mongodb+srv://urujahmedsyed:beabat@cluster0.64q31qj.mongodb.net/resportal")
db = client['resportal']
collection = db['imgs']

model = tf.keras.models.load_model('./predictions/malaria_cnn_model.h5')
model.compile()

def preprocess_image(image):
    image = image.resize((50, 50))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array



@app.route('/predict', methods=['POST'])
def predict():
    print('Entered predict')
    token = request.form.get('token')
    if token:
        tokendata = jwt.decode(token, 'secret123', algorithms="HS256")
    else:
        tokendata = 'hellohello'
    
    file = request.files['image']
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
    image_array = Image.fromarray(img, 'RGB')

    radioval = request.form.get('radioValue')


    pimage = preprocess_image(image_array)
    
    
    prediction = model.predict(pimage)

    print(prediction)
    print('prediction over')
    
    idx_1 = prediction[0]
    print(idx_1)
    result = idx_1[0]
    print(result)
    
    if result >= 0.6 :
        res = 'Parasitized'
    else :
        res = 'Uninfected'
        
    # Insert the image into MongoDB Atlas
    file.seek(0)
    encoded_img = BytesIO(file.read())
    collection.insert_one({'image': encoded_img.getvalue(), 'result': res, 'ground': radioval, 'username': tokendata['uname'], 'type':'image/png'})
    
    return jsonify({'prediction': res})


if __name__ == '__main__':
    app.run(debug=True)
