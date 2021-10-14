# Single-Sample-Per-Person-FR
Firstly, I used the traditional method, and Generative Adversarial Networks (GANs) to expand the number of
samples for training by increasing the intra-class variation.
In the next stage, I used the transfer learning & fine-tuning technique to train the model for classifying the faces
based on the pre-trained model Facenet trained on the VGGFace2 dataset. The dataset I have used to train the
model is the AR dataset which I generated in the first stage, including 100 classes. I used
Average_Hausdorff_Distance as a loss function stead of CrossEntropy().
After that, I used KNN and SVM classifier to classify with X_train, X_test are 100 features vector from CNNs
when passing through 100 images respectively.
