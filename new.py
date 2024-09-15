import pandas as pd
import requests
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr 
import re
from io import BytesIO

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define the regex pattern for finding numbers followed by units
unit_patterns = {
    'item_weight': re.compile(r'(\d+\.?\d*)\s?(microgram|milligram|gram|kilogram|ounce|pound|ton|mg|g|kg|lb|oz)', re.IGNORECASE),
    'item_volume': re.compile(r'(\d+\.?\d*)\s?(microlitre|millilitre|centilitre|decilitre|litre|cubic foot|cubic inch|fluid ounce|gallon|imperial gallon|pint|quart|cup|ml|l|oz|gal|cl|dl|µl)', re.IGNORECASE),
    'depth': re.compile(r'(\d+\.?\d*)\s?(millimetre|centimetre|metre|inch|"|foot|yard|mm|cm|m|in|ft|yd)', re.IGNORECASE),
    'height': re.compile(r'(\d+\.?\d*)\s?(millimetre|centimetre|metre|inch|"|foot|yard|mm|cm|m|in|ft|yd)', re.IGNORECASE),
    'width': re.compile(r'(\d+\.?\d*)\s?(millimetre|centimetre|metre|inch|"|foot|yard|mm|cm|m|in|ft|yd)', re.IGNORECASE),
    'voltage': re.compile(r'(\d+\.?\d*)\s?(millivolt|kilovolt|volt|mv|kv|v)', re.IGNORECASE),
    'wattage': re.compile(r'(\d+\.?\d*)\s?(kilowatt|watt|kw|w)', re.IGNORECASE),
    'maximum_weight_recommendation': re.compile(r'(\d+\.?\d*)\s?(microgram|milligram|gram|kilogram|ounce|pound|ton|mg|g|kg|lb|oz)', re.IGNORECASE)
}

# Define a mapping from short forms to full forms
unit_conversion_map = {
    'mg': 'milligram',
    'g': 'gram',
    'kg': 'kilogram',
    'lb': 'pound',
    'oz': 'ounce',
    'ml': 'millilitre',
    'l': 'litre',
    'cm': 'centimetre',
    'mm': 'millimetre',
    'in': 'inch',
    '"': 'inch',  # Handle double quotes for inch
    'ft': 'foot',
    'v': 'volt',
    'w': 'watt'
}

# Function to convert short form to full form
def convert_unit_to_full_form(text):
    # Fixing the decimal point issue
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Join separated decimal parts

    # Replace double quotes with "inch"
    text = re.sub(r'"', 'inch', text)

    # Split the text into number and unit
    match = re.match(r'(\d+\.?\d*)\s?(\w+)', text)
    if match:
        number, unit = match.groups()
        full_unit = unit_conversion_map.get(unit.lower(), unit)  # Get full form or return the same unit
        return f"{number} {full_unit}"
    return text

# Function to extract the required attribute from text
def extract_attribute(bounds, attribute):
    # Check if the given attribute has a corresponding pattern
    if attribute not in unit_patterns:
        return "Invalid attribute provided."

    pattern = unit_patterns[attribute]

    # Iterate through the detected text and find the first valid unit
    for bound in bounds:
        # Convert the detected text to lowercase
        text = bound[1].lower()  # Convert to lowercase for case-insensitive matching

        # Search for the pattern in the current text
        match = pattern.search(text)
        if match:
            extracted_value = match.group(0)  # Extract the number and unit
            return convert_unit_to_full_form(extracted_value)  # Convert the unit to full form

    return "Attribute not found"

# Function to analyze and enhance the image based on its properties
def adaptive_preprocessing(img):
    img_np = np.array(img)
    avg_brightness = np.mean(img_np)

    # If the brightness is low, increase it
    if avg_brightness < 100:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.5)  # Increase brightness by 1.5x
    elif avg_brightness > 200:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)  # Decrease brightness slightly

    # Optionally apply contrast based on brightness levels
    if avg_brightness < 120:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)

    # Use edge detection to determine if sharpening is necessary
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    edge_intensity = np.mean(edge_array)

    # If edges are not strong, increase sharpness
    if edge_intensity < 50:
        img = img.filter(ImageFilter.SHARPEN)
        for _ in range(2):  # Sharpen multiple times if necessary
            img = img.filter(ImageFilter.SHARPEN)

    return img

# Function to correct misreads like "140omg" where 'o' is detected instead of '0'
def correct_misread(text):
    # Replace 'o' with '0' when it follows a numeric character
    corrected_text = re.sub(r'(\d)o', r'\1o', text)  # Adjust as necessary
    return corrected_text

# Example usage
def imgtotext(image_url, attribute):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Preprocessing Steps
    img = img.convert('L')  # Grayscale conversion
    
    img = adaptive_preprocessing(img)
    # Convert the image to a numpy array
    img_np = np.array(img)

    # Perform text detection using easyocr
    bounds = reader.readtext(img_np)
    
    # Convert all elements in bounds to lowercase and apply the misread correction
    bounds = [(bound[0], correct_misread(bound[1].lower()), bound[2]) for bound in bounds]

    # Extract the attribute
    return extract_attribute(bounds, attribute)

# Read the datasets
dataset = pd.read_csv('train.csv')

# Select only the first 10 rows for testing
dataset_subset = dataset.head(1)

# Initialize lists to store predictions and actual values
predictions = []
actuals = dataset_subset.iloc[:, 3].tolist()  # Assuming the second column contains actual outputs

# Iterate over the rows of the subset
for index, row in dataset_subset.iterrows():
    image_url = row.iloc[0]  # Image link
    attribute = row.iloc[2]  # Attribute (4th column)
    
    # Predict the attribute value
    prediction = imgtotext(image_url, attribute)
    predictions.append(normalize_measurement(prediction))

# Calculate accuracy
correct_predictions = sum(p == a for p, a in zip(predictions, actuals))
accuracy = correct_predictions / len(actuals) if actuals else 0

print(f'Accuracy: {accuracy * 100:.2f}%')
