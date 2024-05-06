"""
This file will create the YOLO model object from pytorch.
"""
import torch
import torch.nn as nn
import os
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

class YOLO:
    def __init__(self, model_path, grid_size, num_boxes, num_classes):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Initialize parameters
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # Load model if it exists, otherwise create a new one
        if os.path.exists(model_path):
            logging.info("Model loading not implemented yet.")
            # self.model = self.load_model(model_path)  # Placeholder for actual load function
        else:
            logging.error("Model path does not exist, creating a new model.")
            self.create_model()

    def create_model(self):
        """Create a model architecture suitable for processing Full HD images."""
        self.features = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, stride=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Reducing to 960x540

        nn.Conv2d(16, 32, 3, padding=1, stride=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Reducing to 480x270

        nn.Conv2d(32, 64, 3, padding=1, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Reducing to 240x135

        nn.Conv2d(64, 128, 3, padding=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Reducing to 120x67

        nn.Conv2d(128, 256, 3, padding=1, stride=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Reducing to 60x33

        nn.Conv2d(256, 512, 3, padding=1, stride=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # Reducing to 30x16

        nn.Conv2d(512, 1024, 3, padding=1, stride=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, padding=1),  # Reducing to 15x8
        ).to(self.device)

        self.classifier = nn.Sequential(
        nn.Conv2d(1024, 1024, 3, padding=1, stride=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.Conv2d(1024, self.grid_size * self.grid_size * (self.num_classes + self.num_boxes * 5), 1),
    ).to(self.device)

    def forward(self, x: torch.Tensor):
        """Forward pass through the YOLO model."""
        x = self.features(x)
        x = self.classifier(x)
        return x.reshape(-1, self.grid_size, self.grid_size, self.num_classes + self.num_boxes * 5)

# Example of how to create a YOLO object
yolo_model = YOLO("path/to/model.pth", 7, 2, 20)
