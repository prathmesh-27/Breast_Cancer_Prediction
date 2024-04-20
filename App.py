
from flask import Flask,request,render_template
import os
import joblib
import pandas as pd

app = Flask(__name__)

def predict_classes(input_data):
    model_path= os.getcwd()+r'\models\model'

    std,pca,classifer = joblib.load(model_path+r'\cancer_predictor.pkl')
    input_data=std.transform(input_data)
    input_data = pca.transform(input_data)
    prediction= classifer.predict(input_data)
    return prediction[0]


@app.route('/')
def index():
    return render_template('home.html')        

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        mean_texture = float(request.form['mean_texture'])
        mean_smoothness = float(request.form['mean_smoothness'])
        mean_concavity = float(request.form['mean_concavity'])
        mean_concave_points = float(request.form['mean_concave_points'])
        mean_symmetry = float(request.form['mean_symmetry'])
        mean_fractal_dimension = float(request.form['mean_fractal_dimension'])
        radius_error = float(request.form['radius_error'])
        texture_error = float(request.form['texture_error'])
        perimeter_error = float(request.form['perimeter_error'])
        area_error = float(request.form['area_error'])
        smoothness_error = float(request.form['smoothness_error'])
        compactness_error = float(request.form['compactness_error'])
        concavity_error = float(request.form['concavity_error'])
        concave_points_error = float(request.form['concave_points_error'])
        symmetry_error = float(request.form['symmetry_error'])
        fractal_dimension_error = float(request.form['fractal_dimension_error'])
        worst_radius = float(request.form['worst_radius'])
        worst_texture = float(request.form['worst_texture'])
        worst_perimeter = float(request.form['worst_perimeter'])
        worst_area = float(request.form['worst_area'])
        worst_smoothness = float(request.form['worst_smoothness'])
        worst_compactness = float(request.form['worst_compactness'])
        worst_concavity = float(request.form['worst_concavity'])
        worst_concave_points = float(request.form['worst_concave_points'])
        worst_symmetry = float(request.form['worst_symmetry'])
        worst_fractal_dimension = float(request.form['worst_fractal_dimension'])

        # Create a DataFrame with the input data
        data = pd.DataFrame({
            'mean texture': [mean_texture],
            'mean smoothness': [mean_smoothness],
            'mean concavity': [mean_concavity],
            'mean concave points': [mean_concave_points],
            'mean symmetry': [mean_symmetry],
            'mean fractal dimension': [mean_fractal_dimension],
            'radius error': [radius_error],
            'texture error': [texture_error],
            'perimeter error': [perimeter_error],
            'area error': [area_error],
            'smoothness error': [smoothness_error],
            'compactness error': [compactness_error],
            'concavity error': [concavity_error],
            'concave points error': [concave_points_error],
            'symmetry error': [symmetry_error],
            'fractal dimension error': [fractal_dimension_error],
            'worst radius': [worst_radius],
            'worst texture': [worst_texture],
            'worst perimeter': [worst_perimeter],
            'worst area': [worst_area],
            'worst smoothness': [worst_smoothness],
            'worst compactness': [worst_compactness],
            'worst concavity': [worst_concavity],
            'worst concave points': [worst_concave_points],
            'worst symmetry': [worst_symmetry],
            'worst fractal dimension': [worst_fractal_dimension],
        })
        
        # Make prediction
        prediction = predict_classes(data)
        return render_template('home.html', pred=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=8080)