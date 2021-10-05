# nn_image_classifier_pneumonia

## Data Understanding

Our data consists of almost 6,000 thousand chest X-rays images (JPEG) taken of children ages 1-5 in the city of Guangzhou, China. Some images show normal, healthy lungs, while most show evidence of bacterial or viral pneumonia. 

There are over 5,000 images in the training set, while 624 images have been set aside for final evaluation as a test set. The data is fairly imbalanced, with the target "Pneumonia" class over twice as prevalent as the "Normal" class. (In the training set, there are about 1,300 "Normal" lung images, but just under 4,000 "Pneumonia" lung images.) The image sizes are highly diverse; the largest is over 3x the size of the smallest. The Pneumonia class images have a smaller range in size than the Normal class images.

Pneumonia in the lungs presents differently depending on its source (bacterial or viral), but both present as a cloudiness or occlusion on the X-ray image. Bacterial pneumonia tends to consolidate in one area, or lobe, of the lungs, while viral pneumonia tends to be more diffuse. 

One limitation of this dataset is that all the X-ray images are from children. It's possible that our results are not generalizable or our model's success applicable to adult cases of pneumonia.
