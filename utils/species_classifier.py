import pandas as pd
import numpy as np
from pathlib import Path
from torchvision import models
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

class SpeciesClassifier:
    @classmethod
    def build_species_classifier(cls, classifier_path):
        if not classifier_path:
            return None
        else:
            return cls(classifier_path)

    def __init__(self, classifier_path: str):
        # Hyperparameters
        self.confidence_threshold = 0.45
        
        classifier_path = Path(classifier_path)
        model_path = classifier_path / "classifier.pt"
        class_mapping_path = classifier_path / "class_mapping.txt"
        assert model_path.exists(), f"Invalid classifier path {classifier_path}"
        assert class_mapping_path.exists(), f"Invalid class_mapping path {classifier_path}"

        with open(str(class_mapping_path), "r") as f:
            self.classes = f.readline().strip().split(",")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Initializing classifier for species {self.classes} on device {self.device}")        

        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, len(self.classes))

        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.model.eval()
        
        # Transformations to make the input valid to the model
        average_patch_size = (200, 400)
        self.transform = v2.Compose([
            v2.Resize(average_patch_size),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __call__(self, row: pd.Series, image: np.array) -> str:
        xmin = int(row["xmin"])
        xmax = int(row["xmax"])
        ymin = int(row["ymin"])
        ymax = int(row["ymax"])

        patch = image[ymin:ymax, xmin:xmax]
        patch = self.transform(patch)

        # Add batch dimension
        patch = patch.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(patch)
            outputs = F.softmax(outputs, dim=1)
            
            confidences, preds = torch.max(outputs, 1)

        predicted_class = self.classes[preds.item()]
        confidence = confidences.item()
        if confidence < self.confidence_threshold:
            predicted_class = None
        return confidence, predicted_class
