Image size given - 101x101
upsampled to 128x128

Training examples given - 4000

Divided into training and validation based on stratified sampling based on percentage of salt in each image

Training images after augmentation - 3200 x 4 = 12800

Augmentation techniques used :
Horizontal | Vertical | Horizontal-Vertical flip


Learning rate:
lr=0.001
min_lr=0.000001

Batch size :
4 - small enough to fit into RAM (12 GB) | big enough to update weights at this rate


Optimization:
RMSProp (dampen oscillations, choses a different learning rate for each parameter)


Loss:
last layer sigmoid activation(0-1) followed by Cross-entropy loss

BatchNorm layer

Dropout layer - 0.25 and 0.5

ResNet layer


