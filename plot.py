import torch
from model import CNNModel
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

# Define model
model = CNNModel(num_classes=8).cuda()

# Create a SummaryWriter
writer = SummaryWriter('plots/')

# Dummy input to visualize the graph
dummy_input = torch.zeros(1, 3, 256, 256).cuda()

# Add model graph to TensorBoard
writer.add_graph(model, dummy_input)
writer.close()
