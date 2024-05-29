import torch
import cv2
import torchvision.transforms as transforms
import argparse
from model import CNNModel

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
    default='C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/test/Mixed/744980_ss_1b894e992d8bdd14d4d40350afa3d20e52799cec.1920x1080.jpg',
    help='C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/test/')
args = vars(parser.parse_args())


# the computation device
device = ('cuda')
# list containing all the class labels
labels = [
    "Mixed","Mostly Negative","Mostly Positive","Negative","Overwhelmingly Positive","Positive","Very Positive","Very Negative"
    ]

# initialize the model and load the trained weights
model = CNNModel(50).to(device)
checkpoint = torch.load('C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/old model/model 2/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# read and preprocess the image
image = cv2.imread(args['input'])
# get the ground truth class
gt_class = args['input'].split('/')[-2]
orig_image = image.copy()
# convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]
cv2.putText(orig_image, 
    f"GT: {gt_class}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6, (0, 255, 0), 2, cv2.LINE_AA
)
cv2.putText(orig_image, 
    f"Pred: {pred_class}",
    (10, 55),
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6, (0, 0, 255), 2, cv2.LINE_AA
)
print(f"GT: {gt_class}, pred: {pred_class}")
cv2.imshow('Result', orig_image)
cv2.waitKey(0)
cv2.imwrite(f"outputs/{gt_class}{args['input'].split('/')[-1].split('.')[0]}.png",
    orig_image)
