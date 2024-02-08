import os
import shutil
from sklearn.model_selection import train_test_split

# Set the paths to your image and label files
image_dir = '/home/eyerov/Documents/meera/screenshots/img'
label_dir = '/home/eyerov/Documents/meera/screenshots/lab'

# Get the list of image and label files
image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Make sure the lists are sorted to match images with labels
image_files.sort()
label_files.sort()

# Split the data into train, test, and validation sets
# Adjust the test_size and validation_size based on your needs
image_train, image_test, label_train, label_test = train_test_split(image_files, label_files, test_size=0.2, random_state=42)
image_train, image_val, label_train, label_val = train_test_split(image_train, label_train, test_size=0.25, random_state=42)

# Function to move files to their respective directories
def move_files(src_dir, dst_dir, file_list):
    os.makedirs(dst_dir, exist_ok=True)  # Create destination directory if it doesn't exist
    for file in file_list:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)
        shutil.move(src_path, dst_path)

# Create train, test, and validation directories
train_dir = '/home/eyerov/Documents/meera/screenshots/train'
test_dir = '/home/eyerov/Documents/meera/screenshots/test'
val_dir = '/home/eyerov/Documents/meera/screenshots/validate'

# Move files to their respective directories
move_files(image_dir, os.path.join(train_dir, 'img'), image_train)
move_files(label_dir, os.path.join(train_dir, 'lab'), label_train)

move_files(image_dir, os.path.join(test_dir, 'img'), image_test)
move_files(label_dir, os.path.join(test_dir, 'lab'), label_test)

move_files(image_dir, os.path.join(val_dir, 'img'), image_val)
move_files(label_dir, os.path.join(val_dir, 'lab'), label_val)

