import os
from PIL import Image

# Specify the paths for your source and destination folders
source_folder = r'C:\Users\hiche\Desktop\PFE EX\train\Tomato_Yellow_Leaf_Curl_Virus'  # Replace with your source folder path
destination_folder = r'C:\Users\hiche\Desktop\PFE EX\train 1\TomatoTomato_Yellow_Leaf_Curl_Virus'  # Replace with your destination folder path

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Get a list of all JPEG files in the source folder
jpeg_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]

# Convert each JPEG file to PNG and save it in the destination folder
for jpeg_file in jpeg_files:
    try:
        image = Image.open(os.path.join(source_folder, jpeg_file))
        png_file = os.path.splitext(jpeg_file)[0] + '.png'
        image.save(os.path.join(destination_folder, png_file), 'PNG')
        print(f"Converted {jpeg_file} to {png_file}")
    except Exception as e:
        print(f"Error converting {jpeg_file}: {e}")
