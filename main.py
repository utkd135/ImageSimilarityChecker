from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity


dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').cuda()
app = Flask(__name__, template_folder='C:/Users/utkar/Desktop/Factory_project/webApp/templates/')
run_with_ngrok(app)  # Comment this line if you're not using ngrok

# Define your transformations and models here
# transform = transforms.Compose([
#     transforms.CenterCrop(800),
#     transforms.Resize(448),
#     transforms.ToTensor(),
#     transforms.Normalize(0.5, 0.5)
# ])

def preprocess_transform(img_path, degree=0):
    transform = transforms.Compose([
        transforms.CenterCrop(1000),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    # image = Image.open(img_path)
    image = transforms.functional.rotate(img_path, degree)
    return transform(image).unsqueeze(0)

# Step 4: Feature Extraction
def extract_features(img):
    """
    args
    img : image of the ornament
    """
    img_tensor0 = preprocess_transform(img, 0)
    img_tensor1 = preprocess_transform(img, 90)
    img_tensor2 = preprocess_transform(img, 180)
    img_tensor3 = preprocess_transform(img, 270)
    with torch.no_grad():
        feature1 = dinov2_vits14_reg(img_tensor0.cuda())
        feature2 = dinov2_vits14_reg(img_tensor1.cuda())
        feature3 = dinov2_vits14_reg(img_tensor2.cuda())
        feature4 = dinov2_vits14_reg(img_tensor3.cuda())
    features = (feature1 + feature2 + feature3 + feature4)/4
    return features

def upload_image(code_name, image, type):
    pil_image = Image.fromarray(np.array(image))
    base_path = r'C:/Users/utkar/Desktop/Factory_project/webApp/'

    # Check for path
    if not os.path.exists(path=f"{base_path}{type}"):
        os.mkdir(f"{base_path}{type}")
    if not os.path.exists(path=f"{base_path}{type}_feature"):
        os.mkdir(f"{base_path}{type}_feature")

    # Save the image
    save_path = f"{base_path}{type}/{code_name}.png"
    pil_image.save(save_path)

    # Correspondingly extract features and save in a directory
    features = extract_features(pil_image).cpu()

    # Normalize the vector to unit length
    features /= features.norm()

    feature_path = os.path.join(base_path, f'{type}_feature/{code_name}_feature.pt')
    torch.save(features, feature_path)

    return "Image saved"

global_feature_database = r'C:/Users/utkar/Desktop/Factory_project/webApp/ornament_feature/'
global_image_database = r'C:/Users/utkar/Desktop/Factory_project/webApp/ornament/'

def find_design_number(image, type):
    # Preprocessing
    image = Image.fromarray(np.array(image))
    image_f = extract_features(image)[0]
    image_f = image_f.cuda()
    
    # Load all features into a single tensor
    feature_database_path = global_feature_database.replace('ornament', type)
    all_features = torch.stack([torch.load(os.path.join(feature_database_path, feature)) for feature in os.listdir(feature_database_path)])
    # Normalize vectors to unit length

    image_f /= image_f.norm()
    all_features /= all_features.norm(dim=2, keepdim=True)

    # Compute cosine similarity using torch.matmul for batched computation
    cosine_similarities = torch.matmul(all_features.cuda(), image_f).cpu()
    
    # Find the maximum similarity and corresponding design number
    max_sim, idx = cosine_similarities.max(dim=0)
    max_sim = max_sim.item()
    design_number = os.listdir(feature_database_path)[idx]

    # Load the corresponding image
    img = Image.open(os.path.join(global_image_database.replace('ornament', type), design_number.replace('_feature.pt', '.png')))

    return max_sim, design_number.replace('_feature.pt', ''), img
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image_route():
    code_name = request.form['code_name']
    image = request.files['image']
    type = request.form['type']

    # Save the uploaded image and extract features
    result = upload_image(code_name, Image.open(image), type)
    return render_template('result.html', result=result)

@app.route('/get_design_number', methods=['POST'])
def get_design_number_route():
    image = request.files['image']
    type = request.form['type']

    # Get design number and similarity score
    similarity, design_number, img = find_design_number(Image.open(image), type)
    return render_template('result.html', similarity=similarity, design_number=design_number, img=img)

if __name__ == '__main__':
    app.run()
