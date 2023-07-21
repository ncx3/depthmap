import torch
import torchvision.transforms as transforms
import cv2
import urllib
import os

# Set the TORCH_HOME environment variable to change the download directory
os.environ['TORCH_HOME'] = 'D:\\torch_cache'

# Load the MiDaS model from Torch Hub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Transformation to apply to the input image before passing it to the model
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(384),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the image
img_path = "image_path.jpg"  # replace with your image path
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply the transformation, add a batch dimension, and move to the device
input_img = transform(img).unsqueeze(0).to(device)

# Predict the depth map
with torch.no_grad():
    prediction = midas(input_img)

# Move the depth map to CPU and transform it to a numpy array
depth_map = prediction.squeeze().cpu().numpy()

# Normalize the depth map for visualization
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# Define output path (same directory as input image)
output_path = os.path.join(os.path.dirname(img_path), "depth_map.png")

# Save the depth map
cv2.imwrite(output_path, (depth_map * 255).astype('uint8'))
