# chest-X-Ray-Images-classification
a simple CNN  achieve best performence on Kaggle dataset  chest X-Ray Images(Penumonia)


a 7 layers convolutional neural network with 1.72million prarameters achieves best performance on dataset from Kaggle Chest X-Ray Images(Pneumonia) , dataset can be accessed from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.

to handle the imbalanced training datasets(normal:1341,penumonia:3874), we generate new negtive samples using ImageDataGenerator() from keras,this improve precision a lot. Another tactics is mutiply the classifying loss function by a weight conffient,this also helpful to improve the performance of the network.

## Enviroments
- Python 3.5
- Tensorflow 1.9.0
- keras 2.1.2

## Usage
First, download dataset to your repository.

Config hyperparameters in config.yml file.

To train a model: $ python training.py

After 11000 iterations  recall is 0.897 precision is 0.904 on the test datasets.
