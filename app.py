from flask import Flask, render_template, redirect, flash, session, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yoursecretkey'

# Define the neural network architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 output classes
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% probability

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Applying dropout
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('sjain35_assignment0_part_3.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Define ImageForm
class ImageForm(FlaskForm):
    image = FileField('Image', validators=[DataRequired()])
    submit = SubmitField('Predict')

# Route for uploading an image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = ImageForm()
    if form.validate_on_submit():
        try:
            image_file = form.image.data
            image = Image.open(BytesIO(image_file.read())).convert("L")  # Convert to grayscale
            image = transform(image).unsqueeze(0)   # Apply transformations
            image /= 255.0  # Convert pixel values to the range [0, 1]

            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            # Convert the prediction to a class label
            class_labels = ['choroidal neovascularization', 'diabetic macular edema', 'drusen', 'normal']
            prediction_label = class_labels[predicted.item()]

            session['prediction'] = prediction_label
            flash('Image uploaded successfully')
            return redirect(url_for('predict'))
        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(url_for('upload_image'))

    return render_template('index.html', form=form)

# Route for displaying prediction
@app.route('/predict', methods=['GET'])
def predict():
    prediction_label = session.get('prediction')
    return render_template('predict.html', prediction_label=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
