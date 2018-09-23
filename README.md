# **Training & inference on MNIST** 

---

### Goals

The goals / steps of this project are the following:
* Train a convolutional neural network on mnist dataset
* Run an inference server to make predictions

### Files

The project includes the following files:
* train.py contains script to train on the mnist dataset
* inference_server.py contains script to run the inference server
* test*.png are some test images to try running the inference server

### Training
The network can be trained by running the following: 
```sh
python train.py
```
The following hyperparameters can be passed as argument
1. Dropout probability for all the convolution layers
2. Dropout probability for all the fully connected layers
3. Epochs
4. Batch size

Please execute the following to understand how to input these hyperparameters
```sh
python train.py -h
```

### Inference server
The inference server implementation is inspired by https://github.com/jrosebr1/simple-keras-rest-api.
The keras+flask server can be started by running:
```sh
$ python inference_server.py 
Using TensorFlow backend.
* Loading Keras model and Flask starting server...please wait until server has fully started
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
Requests can be submitted via curl:
```sh
$ curl -X POST -F image=@test0.png 'http://localhost:5000/predict'
{
  "Predictions": [
    {
      "Label": 2, 
      "Softmax probability": 0.9721488356590271
    }
  ], 
  "Success": true
}
```
### Results
The training was run for 10 epochs on a CPU and results are as follows. Note that training set was divided into training (90%) and validation sets (10%)
* training accuracy = 99.04%
* validation accuracy = 99.02%
* test accuracy = 99.21%

I was satisfied with the results on the 1st set of hyperparameters and did not perform too much tuning.