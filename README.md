# Apple-Leaf-Disease-Detection-Project
Alternaria leaf spot, Brown spot,Grey spot, and Rust are four common types of apple leaf diseases that severely affect apple yield. 
However, the existing research lacks an accurate and fast detector of apple diseases for ensuring the healthy development of the apple
industry becuase their is a problem of overfitting of data. So, i am trying to propose a deep learning approach with a new dropblock
regularisation method that is based on improved convolutional neural networks (CNNs) for the early detection of apple leaf diseases.

#ABOUT THE DATASET

The dataset used for this project has been taken from Science Data Bank- Dataset which can be found here 
(https://www.scidb.cn/en/detail?dataSetId=0e1f57004db842f99668d82183afd578).
The data used for this project is situated in the folder named “ATLDSD” in the Github Repository.
The Data fed for the modeling is of Apple Leaves. For training purpose Alternaria leaf spot, Brown spot,Grey spot, 
and Rust are four common types of apple leaf diseases used.

#PROPERTIES OF IMAGES

Type of File : JPG File.
Dimensions : 256 * 256.
Width : 256 Pixels.
Height : 256 Pixels.
Horizontal Resolution : 96 dpi.
Vertical Resolution : 96 dpi.
Bit Depth : 24.

#STEPS INVOLVED

1. Collecting the Dataset: 
  All the pictures of apple leaf diseases were collected from four different apple experimental
  demonstration stations of Northwest University of agriculture and forestry science and technology in Northwest China. 
  These images were taken using glory V10 mobile phone. Apple leaf images with different degrees of disease were taken 
  in the laboratory (about 51.9%) and the real cultivation field (about 48.1%), under different weather conditions and 
  at different times of the day. These diseases include alternating leaf spot, gray spot, brown spot and rust
  https://www.scidb.cn/en/detail?dataSetId=0e1f57004db842f99668d82183afd578.
 
2. Dataset Image Preprocessing:
  Due to powerful end-to-end learning, deep learning models do not require much image
  preprocessing. We apply data augmentation and data normalization during the preprocessing step.

3. Data Augmentation:
  One of the main advantages of CNNs is their ability to generalize, that is, their ability to process
  data that has never been observed. But when the data size is not large enough and the data diversity
  is limited, they tend to over-fit the training data, which means they cannot be generalized.
  In order to enhance the generalization ability of the network and reduce over-fitting, the dataset was
  expanded by data augmentation technology to simulate changes in lighting, exposure, angle, and noise
  during the preprocessing step of apple leaf images.
 
4. Data Normalization:
   Considering that the deep neural network is very sensitive to the input feature range, a large
   range of eigenvalues will cause instability during model training [25]. In order to improve the CNN
   convergence speed and learn subtle differences between images, the dataset is normalized. For each
   image channels, the data are normalized by Equation (1).
   Zi = (xi − x)/δx
  
5. Dividing the Dataset:
  For the training and testing of DCNNs, the dataset with 2970 images were divided into three
  independent subsets, including training dataset, validation dataset, and testing dataset.
  
6. Model building:
  Convolutional Neural Network (CNN), a class of deep learning neural network, popularly used for image 
  classification and processing, is proposed to accomplish this task. CNN, like other neural networks, 
  are made up of neurons with learnable weights and biases. Each neuron receives several nputs in 
  form of raw image matrices, calculates a weighted sum over them passes it through an activation function
  and responds with an output. The whole network has a loss function that reduces gradually with each iteration
  and eventually the network learns a set of parameters which best fit the classification task. The result 
  obtained is a vector of final probabilities for each class.


 
 
