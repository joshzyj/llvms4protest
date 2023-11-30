# An example to infer images on global protests
import os
import pandas as pd
import torch
from torchvision.transforms import functional as F
#from transformers import Swinv2ForImageClassification, AutoImageProcessor
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from transformers import Swinv2ForImageClassification, Swinv2Config, AutoImageProcessor, Trainer, TrainingArguments
# Load the fine-tuned model and processor
model_dir = "./swin_v2_b_model_protest"
model_name = "microsoft/swinv2-base-patch4-window8-256"
config = Swinv2Config.from_pretrained(model_name)
config.num_labels = 2  # Set the number of output classes
model = Swinv2ForImageClassification.from_pretrained(model_dir, config=config,ignore_mismatched_sizes=True)
processor = AutoImageProcessor.from_pretrained(model_name,config=config,do_rescale=False)

# Define a function to predict the class of a single image
def predict_image(image_info):
    country_folder, image_file = image_info
    image_path = os.path.join(image_folder, country_folder, image_file)
    try:
        # Attempt to open the file as an image
        image = Image.open(image_path)
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL image to PyTorch tensor
        image = F.to_tensor(image).float()
        inputs = processor(image, return_tensors="pt")
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
        return {"Country": country_folder, "Image": image_file, "Prediction": prediction.item()}
    except Exception as e:
        print(f"Error processing file {image_path}: {str(e)}")
    return None

# Iterate through the files in the "images_by_country" folder and make predictions
image_folder = "images_by_country"
image_infos = []

for country_folder in os.listdir(image_folder):
    country_path = os.path.join(image_folder, country_folder)
    if os.path.isdir(country_path):
        for image_file in os.listdir(country_path):
            image_infos.append((country_folder, image_file))

# Use ThreadPoolExecutor to parallelize the image prediction
with ThreadPoolExecutor() as executor:
    predictions = list(tqdm(executor.map(predict_image, image_infos), total=len(image_infos), desc="Predicting"))

# Remove None entries (if any)
predictions = [pred for pred in predictions if pred is not None]

# Save the predictions to a CSV file
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("predictions_by_country.csv", index=False)
