from flask import Flask,request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

def model_predict(file_path):
    model=load_model('pneumonia_model.h5')
    img=image.load_img(file_path,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    pred=model.predict(img)
    pred=pred.argmax()
    return pred

@app.route('/predict',methods=['POST'])
def upload():
    
    """Pneumonia Prediction 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Upload The Chest X-Ray
    responses:
        200:
            description: The output values
        
    """
    f=request.files['file']       
    base_path=os.getcwd()
    file_path=os.path.join(base_path,'chest_xray\\val',f.filename)
    
    pred=model_predict(file_path)
    if pred==1:
        return "The person has pneumonia"
    return "The person doesn't have pneumonia"


@app.route('/')
def home():
    return "Welcome"

if __name__=="__main__":
    app.run(debug=False)