# Sonar-Image-Corner-Point-Coordinate-Detection

![marked_image](https://github.com/meerap1/Sonar-Image-Corner-Point-Coordinate-Detection/assets/156745402/5fbd5508-bd56-4b7f-9d11-5fb48634e139)

## Table of Content
1. Introduction
2. Data Annotation
3. Data Augmentation
4. Test-Train-Validate Splitting
5. Training the Model
6. Testing the Model
## Introduction
The project focuses on developing a machine learning model capable of detecting and marking the coordinates of corner points in sonar images. In this project, we aim to implement algorithms that can automatically identify and extract corner points from sonar images.In this project, we aim to implement algorithms that can automatically identify and extract corner points from sonar images. By employing machine learning techniques, we seek to train models to recognize patterns associated with corner points in the image data. These models will be trained on labeled datasets containing sonar images along with the corresponding coordinates of corner points. The trained model will then be able to predict the coordinates of corner points in new, unseen sonar images. Overall, the goal of this project is to develop an efficient and accurate system for corner point coordinate detection in sonar images.
## Data Annotation

I've developed a simple annotation model in Python that enables me to extract the x and y coordinates of points of interest by clicking on images. Initially, I imported the OpenCV library to handle image operations. Subsequently, I defined a function, `mouse_click_event`, which prints out the coordinates whenever a left mouse button click event occurs on the displayed image.

For implementation, I loaded a sample image using the provided file path. The script reads the image using the `cv2.imread` function and displays it using the `cv2.imshow` method. I then defined a named window using `cv2.namedWindow` and set a mouse callback function with `cv2.setMouseCallback` to capture mouse events. Upon clicking on the image, the script prints the coordinates and waits for further user interaction. 
To test the model, I executed the script, inputted the image path, and observed the displayed image. Upon clicking on various points of interest within the image, the console printed out their respective coordinates. This straightforward approach allows for the rapid collection of coordinate data from images, essential for tasks such as corner point detection or object localization. Through this process, I gathered the coordinates of corner points from 40 original images and stored them in text files with the same names as the corresponding images.
## Data Augmentation
I developed an image augmentation model in Python to enhance my dataset for corner point coordinate detection. By augmenting 40 original images, I aimed to increase the diversity in the dataset, accounting for variations in size, color, rotation, and flipping. To begin, I imported necessary libraries such as `os`, `sys`, and `cv2`, enabling image manipulation and file management functionalities. 

With the original images and labels stored in separate folders, I created a new directory to store the augmented data. The augmentation process involved randomly shifting pixels within a specified range to simulate variations. Using the `random_pixel_shift` function, I implemented pixel-level transformations to create new images while preserving the original content. This function efficiently handled pixel shifting, ensuring the augmented images maintained the context of the original scenes.

In the augmentation loop, I iterated through a predefined number of iterations to generate a target number of augmented images. For each iteration, I randomly selected an original image and its corresponding label. Utilizing OpenCV, I loaded the image, applied pixel shifts, and saved the augmented image to the designated folder. Additionally, I adjusted the label coordinates to align with the pixel shifts, ensuring consistency between image and label annotations.

Throughout the process, I visualized the augmented images to validate the effectiveness of the augmentation technique. By displaying the augmented images alongside their corresponding annotations, I verified that the pixel shifts accurately reflected changes in the image coordinates. Once the augmentation process was complete, I had successfully expanded my dataset, providing a more diverse set of images for training my corner point coordinate detection model.
##  Test-Train-Validate Splitting
After augmenting the original dataset, I proceeded to split the data and labels into distinct sets for training, testing, and validation purposes. This step is crucial for ensuring the robustness and generalization of the model. Utilizing the `train_test_split` function from the `sklearn.model_selection` module, I divided the dataset into three subsets: training, testing, and validation, maintaining a ratio of 3:1:1, respectively.

The process involved organizing both the image and label files into their respective directories. Initially, I obtained lists of image and label files from the specified directories, ensuring they were sorted to maintain consistency between images and labels. Subsequently, I split the lists into training, testing, and validation sets using `train_test_split`, specifying the desired ratios and setting a random seed for reproducibility.

With the dataset split, I created new directories for each subset: `train`, `test`, and `validate`. These directories served as the destination for moving the respective image and label files. To facilitate this movement, I defined a custom function, `move_files`, which systematically transferred files from the source directory to their corresponding destination directories.

Finally, I executed the `move_files` function for each subset, relocating the image and label files accordingly. This ensured that the dataset was appropriately partitioned, ready for subsequent model training, evaluation, and validation. By organizing the data into distinct sets, I laid the foundation for robust model development, enabling effective assessment of model performance and generalization capabilities.
## Training the Model
After splitting my dataset, I proceeded to train a more complex convolutional neural network (CNN) model for corner point detection. I defined a custom dataset class, `CornerDataset`, which handled loading images and their corresponding labels from the specified directories. To enhance the diversity of the dataset and improve model robustness, I applied data augmentation techniques such as random resizing and horizontal flipping during training.

The architecture of the CNN model, named `ComplexCornerDetectionModel`, consisted of multiple convolutional layers followed by max-pooling and fully connected layers. This architecture was designed to extract complex features from the input images and predict the x and y coordinates of corner points. I initialized the model, defined the loss function as Mean Squared Error (MSE), and used the Adam optimizer for parameter optimization.

During the training loop, I iterated over multiple epochs, updating the model parameters based on the training data. Despite implementing data augmentation techniques, including random resizing and flipping, the training and validation losses did not decrease significantly. This lack of improvement in loss values suggests that the model may be overfitting to the training data, failing to generalize well to unseen examples.

Overfitting occurs when a model learns to memorize the training data rather than learning the underlying patterns, resulting in poor performance on unseen data. In this case, the model's architecture may be too complex for the given dataset, leading to overfitting. To address this issue, I could consider reducing the model's complexity, adding regularization techniques such as dropout or L2 regularization, or increasing the amount of training data to improve generalization.

Additionally, I attempted to test the trained model on an unknown image. However, without observing a significant decrease in training and validation losses, it's crucial to approach the model's predictions with caution. Evaluating the model's performance on a separate test dataset and analyzing its ability to generalize to unseen examples would provide further insights into its effectiveness.

Finally, I saved the trained model's parameters to a file (`corner_detector_model.pth`) for future use and evaluation. By addressing the overfitting issue and fine-tuning the model's architecture and training process, I aim to improve its performance in accurately detecting corner points in images.
## Testing the Model on images

After training our corner point detection model, we proceeded to test its performance on images from the test dataset. However, upon evaluating the model's predictions, we observed significant discrepancies between the predicted coordinates and the actual coordinates of corner points in the images. This disparity suggests that the model may not have generalized well to unseen data and is exhibiting poor performance during inference.
##Conclusion
In conclusion, the current performance of my corner point detection model indicates a notable disparity between the predicted coordinates and the actual points in test images. This suggests a significant room for improvement to enhance the model's predictive accuracy and ability to generalize effectively. To address these shortcomings, I will focus on refining the training process, adjusting the model architecture, and augmenting the dataset to encompass a wider variety of scenarios.

Beginning with the dataset, I will meticulously review and augment it with additional samples to ensure it adequately represents the diversity of corner point variations present in real-world images. Simultaneously, I will fine-tune the model's architecture and optimize hyperparameters to mitigate overfitting and improve generalization. By iteratively refining the model and leveraging insights gained from testing and validation, I aim to close the performance gap and achieve more accurate corner point predictions across a diverse range of images.
