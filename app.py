from flask import Flask , render_template
from flask import request , redirect , url_for
from predictor import predict_digit_from_array

import pandas as pd

app = Flask(__name__)


test_df = pd.read_csv('datasets/test.csv')

@app.route('/')
def index():
    total=len(test_df)
    image_file = [f"test_images/img_{i}.png" for i in range(500)]
    return render_template("index.html" , total=total , images = image_file)


@app.route('/predict' , methods=['POST'])
def predict():
    image_filename = request.form.get('image_filename')

    filename = image_filename.split('/')[-1]  # gets 'img_23.png'
    index = int(filename.replace('img_', '').replace('.png', ''))


    pixel_array = test_df.iloc[index].values
    prediction = predict_digit_from_array(pixel_array)

    static_image_path = url_for('static', filename=image_filename)

    image_file = [f"test_images/img_{i}.png" for i in range(500)]

    # Return prediction as plain text for AJAX
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return str(prediction)
    
    # Fallback if accessed via form normally (not AJAX)
    return render_template("index.html", image=static_image_path, images=image_file, prediction=prediction)




if __name__=='__main__':
    app.run(debug=True)