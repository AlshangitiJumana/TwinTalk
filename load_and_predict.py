import os
import cv2
import playsound
import torch
from gtts import gTTS
from pytorch_i3d import InceptionI3d

def load_model(dataset='WLASL2000'):
    """
    Load the InceptionI3d model with specified dataset configuration.
    """
    to_load = {
        'WLASL2000': {'logits': 2000, 'path': 'weights/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'}
    }

    model = InceptionI3d(num_classes=400)
    model.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location='cpu'))
    model.replace_logits(to_load[dataset]['logits'])

    if 'path' in to_load[dataset]:
        model.load_state_dict(torch.load(to_load[dataset]['path'], map_location='cpu'), strict=False)

    model.eval()
    model.cpu()
    return model

# def classify_live(input_tensor, model, idx2label, start_ignore=12, threshold=0.5):
#     """
#     Classify the live input tensor using the WLASL model and return the top predicted word.
#     """
#     with torch.no_grad():
#         per_frame_logits = model.forward(input_tensor)  # ✅ Use forward() explicitly

#     predictions = torch.mean(per_frame_logits, dim=2)
#     _, top_indices = torch.topk(predictions, 1)  # Just need the top prediction
#     top_index = top_indices.cpu().numpy()[0][0]

#     predictions = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()[0]
#     top_prediction = idx2label[top_index]
#     top_prediction_confidence = predictions[top_index]

#     # ✅ Directly use the model's prediction (no LLM logic)
#     word_to_speak = top_prediction

#     # ✅ Convert the chosen word to speech
#     tts = gTTS(text=word_to_speak, lang='en')
#     tts_file = 'prediction.mp3'
#     tts.save(tts_file)
#     playsound.playsound(tts_file)
#     os.remove(tts_file)

#     return word_to_speak  # ✅ Return the predicted word directly

def classify_live(input_tensor, model, idx2label, start_ignore=12, threshold=0.5):
    """
    Classify the live input tensor using the WLASL model and return the top predicted word and confidence score.
    """
    with torch.no_grad():
        per_frame_logits = model.forward(input_tensor)  # ✅ Use forward() explicitly

    predictions = torch.mean(per_frame_logits, dim=2)
    _, top_indices = torch.topk(predictions, 1)  # ✅ Get top prediction index
    top_index = top_indices.cpu().numpy()[0][0]

    probabilities = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()[0]
    top_prediction = idx2label[top_index]
    top_prediction_confidence = probabilities[top_index]  # ✅ Extract confidence score
    #  # ✅ Convert the chosen word to speech
    # tts = gTTS(text=top_prediction, lang='en')
    # tts_file = 'prediction.mp3'
    # tts.save(tts_file)
    # playsound.playsound(tts_file)
    # os.remove(tts_file)

    # ✅ Return both the predicted word & confidence score
    return top_prediction, top_prediction_confidence