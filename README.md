Sports Image Classification AI

A Deep Learning project that classifies sports images using a MobileNetV2-based Convolutional Neural Network.
The model is deployed using a Gradio interface on Hugging Face Spaces for real-time predictions.

Live Demo

https://huggingface.co/spaces/YoussefAlaaa/sports-image-classifier

Project Overview

This project uses Transfer Learning with MobileNetV2 to classify images into 7 sports categories.
It takes an image as input and returns:

Predicted sport class
Confidence score
Top 3 predictions
Classes
Badminton
Cricket
Karate
Soccer
Swimming
Tennis
Wrestling
Model Architecture
Base Model: MobileNetV2 pretrained on ImageNet
Input Shape: 224 × 224 × 3
Global Average Pooling
Dense Layer (128 neurons)
Softmax Output Layer
Loss Function: Categorical Crossentropy
Optimizer: Adam
Tech Stack
Python
TensorFlow / Keras
Gradio
NumPy
Pillow
Hugging Face Spaces
Features
Image upload interface
Real-time prediction
Confidence score output
Top 3 predictions
Clean web interface
Fully deployed AI application
How to Run Locally
git clone https://github.com/your-username/sports-image-classifier.git
cd sports-image-classifier

pip install -r requirements.txt
python app.py
Project Structure
sports-image-classifier/
│
├── app.py
├── sports_mobilenet.keras
├── requirements.txt
└── README.md
Example Output

Input: Image of a sport

Output:
Predicted Class: Soccer
Confidence: 99.75%

Top 3:

Soccer: 99.75%
Wrestling: 0.15%
Tennis: 0.08%
Future Improvements
Add Grad-CAM visualization
Improve model accuracy with fine-tuning
Deploy Flask API version
Improve UI design
Add mobile-friendly interface
Author

Youssef Alaa
AI and Machine Learning Enthusiast
Focused on Computer Vision and Deep Learning
