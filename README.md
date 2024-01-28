# FinetuningExample

[Shortcut](https://github.com/crimson206/FinetuningExample/blob/main/src/fine_tune/script.ipynb) to Script.

## Overview

This repository is designed to provide a simple example of fine-tuning.

## Pretrained Model

The pretrain model used in this example is the ResNet18, available for download from [Torch Hub](https://pytorch.org/hub/pytorch_vision_resnet/).

This model was trained using the [ImageNet1k dataset](https://huggingface.co/datasets/imagenet-1k), known for its diversity and large quantity of images. The ImageNet1k dataset contains 1,000 classes and a total of approximately 1.28 million training images.

## Target Dataset

For fine-tuning, the [Caltech101 dataset](https://data.caltech.edu/records/mzrjq-6wc02) was utilized.

It consists of 101 (or 102) classes with a total of approximately 6,000 images.

## Strategy

1. **Task Adaptation**
   The pretrained model was originally trained for image classification. Therefore, it will also be used to perform a classification task on the new dataset.

   To adapt it to the new task, the final fully connected (fc) layer of the model, originally sized for 1,000 classes, is replaced with a new one appropriate for 102 classes (1000 => 102).

2. **Layer-wise Learning Rate Adjustment**

   - The layers positioned towards the front of the model are more susceptible to changes in weights due to their exposure to gradients from the layers located at the back. However, due to the phenomenon of gradient vanishing, these changes might be gradual. Consequently, the initial layers are generally trained to capture the broad, general features of the given datasets.

   - The layers positioned towards the back have better access to label information and are less influenced by other layers. This positioning enables them to learn more class-specific features.

   - Based on these characteristics, the strategy is to preserve the general features learned by the early layers, which have been exposed to a wide range of images, while actively retraining the layers at the back to focus on the specific characteristics of the newly introduced classes.

   - This approach involves using different learning rates for different layers, specifically [1e-3, 1e-4, 1e-5, 1e-6], to facilitate this targeted learning.

## Training and Early Stopping

Given that this is a transfer learning scenario, training can often be completed more quickly. With this in mind, a relatively small patience value of 3 was employed. The model reached its minimum validation loss after just 5 epochs, and subsequent additional 3 epochs revealed clear signs of overfitting.

## Validation

Finally, to evaluate the newly trained model's performance, the predicted class names along with the actual images were displayed.

![Validation](src/sample_images/output.png)


## Script

For more details, please refer to the link below.

https://github.com/crimson206/FinetuningExample/blob/main/src/fine_tune/script.ipynb