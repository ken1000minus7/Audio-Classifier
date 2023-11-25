# Audio-Classification
We have created an audio classifier using 110 voice recordings - 60 Neutral audio files and 50 Angry audio files.

## Architecture

We used a sequential model so the model can be built layer by layer.

It consists of three layers - an input layer, a hidden layer and an output layer. 

The first layer will receive the input shape. As each sample contains 40 MFCCs (or columns) we have a shape of (1x40) this means we will start with an input shape of 40.

The first two layers will have 256 nodes. The activation function we will be using for our first 2 layers is the ReLU, or Rectified Linear Activation. This activation function has been proven to work well in neural networks.

A Dropout value of 50% will be applied on the first two layers. This will remove random nodes from each cycle, to reduce the likeliness of overfitting.

The output layer will have 2 nodes which matches the number of possible classifications.


### Training the model

Starting with 100 epochs which is the number of times the model will cycle through the data. 
Also beginning with a low batch size, as having a large batch size can reduce the generalisation ability of the model.

```python
from datetime import datetime 

num_epochs = 100
num_batch_size = 32
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test))


duration = datetime.now() - start
print("Training completed in time: ", duration)
```

```python
# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])
```


That's it! Now, we can test it with sample data and get our own personal bit of code which can tell whether a person is angry or not.


### Team
- Manjot Singh Oberoi 20103075
- Rijul Singla 20103080
- Aayush Singh Panwar 20103091
- Armaan Badhan 20103102