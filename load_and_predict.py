import torch
from pytorch_i3d import InceptionI3d  # Ensure you have this module installed
import os
import numpy as np
import torch.nn.functional as F
from utils import load_class_names 

def load_model(dataset='WLASL2000'):
    """
    Load the InceptionI3D model with pre-trained weights.
    """
    weight_paths = {
        'WLASL2000': 'weights/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    }

    # Check if weights exist
    if dataset not in weight_paths or not os.path.exists(weight_paths[dataset]):
        raise FileNotFoundError(f"Weight file {weight_paths[dataset]} not found!")

    # Load model with default 400-class ImageNet configuration
    model = InceptionI3d(num_classes=400)

    # Load ImageNet pre-trained weights first
    model.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location='cpu'))
    
    # Modify model to match the WLASL dataset class size (2000 classes)
    model.replace_logits(2000)  

    # Load ASL-specific weights
    model.load_state_dict(torch.load(weight_paths[dataset], map_location='cpu'), strict=False)

    model.eval()
    model.cpu()
    
    print("Model successfully loaded!")
    return model
def predict_live(model, input_tensor, idx2label):
    """
    Runs the model on the input tensor and returns the predicted sign.
    """
    with torch.no_grad():
        per_frame_logits = model(input_tensor)

    # Average over time (Temporal pooling)
    predictions = torch.mean(per_frame_logits, dim=2)

    # Apply softmax to get confidence scores
    probabilities = F.softmax(predictions, dim=1).cpu().numpy()[0]

    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]  # Get highest probabilities
    top_labels = [idx2label[idx] for idx in top_indices]
    top_confidences = [probabilities[idx] for idx in top_indices]

    return list(zip(top_labels, top_confidences))  # Return (label, confidence) pairs

if __name__ == "__main__":
    model = load_model()
    idx2label = load_class_names()  # Load class names from utils.py

    # Example test with a dummy tensor (simulating a processed video)
    dummy_input = torch.randn(1, 3, 64, 224, 224)  # (Batch, Channels, Frames, Height, Width)
    predictions = predict_live(model, dummy_input, idx2label)
    
    print("Predictions:", predictions)

