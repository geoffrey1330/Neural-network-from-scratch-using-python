# Neural-network-from-scratch-using-python




A machine learning model uses lots of examples to learn the correct weights and bias to assign to each feature in a dataset to help it correctly predict outputs. Back to our proposed solution. You now know that every feature in our dataset must be assigned a weight and that after doing a weighted sum, you add a bias term. In the code block below, you’ll create your neural network class and initialize those weights and biases:

First, you create a neural network class, and then during initialization, you created some variables to hold intermediate calculations. The argument layers is a list that stores your network’s architecture. You can see that it accepts 13 input features, uses 8 nodes in the hidden layer (as we noted earlier), and finally uses 1 node in the output layer. We’ll talk about the other parameters such as the learning rate, sample size and iterations in later sections.




```py
class NeuralNet(): ''' A two layer neural network '''
def __init__(self, layers=[2,5,1], learning_rate=0.001, iterations=100):
    self.params = {}
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.loss = []
    self.sample_size = None
    self.layers = layers
    self.X = None
    self.y = None

def init_weights(self):
    '''
    Initialize the weights from a random normal distribution
    '''
    np.random.seed(1) # Seed the random number generator
    self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) 
    self.params['b1']  =np.random.randn(self.layers[1],)
    self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
    self.params['b2'] = np.random.randn(self.layers[2],)
```




First, i create a neural network class, and then during initialization, i created some variables to hold intermediate calculations. The argument layers is a list that stores your network’s architecture.it accepts 2 input features, uses 5 nodes in the hidden layer (as we noted earlier), and finally uses 1 node in the output layer. We’ll talk about the other parameters such as the learning rate, sample size and iterations in later sections. Moving on to the next code section, i created a function (init_weights) to initialize the weights and biases as random numbers. These weights are initialized from a uniform random distribution and saved to a dictionary called params.





You’ll notice that there are two weight and bias arrays. The first weight array (W1) will have dimensions of 2 by 5—this is because we have 2 input features and 5 hidden nodes, while the first bias (b1) will be a vector of size 8 because you have 8 hidden nodes.





The second weight array (W2) will be a 10 by 1-dimensional array because you have 10 hidden nodes and 1 output node, and finally, the second bias (b2) will be a vector of size because you have just 1 output.





ReLU (Rectified Linear Unit) is a simple function that compares a value with zero. That is, it will return the value passed to it if it is greater than zero; otherwise, it returns zero. The code for the ReLU function is shown below:




```py
def relu(self,Z):
    '''
    The ReLu activation function is to performs a threshold
    operation to each input element where values less 
    than zero are set to zero.
    '''


    return np.maximum(0,Z)
```



In summary, the hidden layer receives values from the input layer, calculates a weighted sum, adds the bias term, and then passes each result through an activation function—in our case a ReLU. The result from the ReLU is then passed to the output layer, where another weighted sum is performed using the second weights and biases. But then instead of passing the result through another activation function, it is passed through what I like to call the output function. The output function will depend on what you’re trying to predict. You can use a sigmoid function when you have a two-class problem (binary classification), and we can use a function called softmax for multi-class problems. In this project,i used a sigmoid function for the output layer. This is because i am predicting one of two classes.





The sigmoid function takes a real number and squashes it to a value between 0 and 1. In other words, it outputs a probability score for every real number. This is useful for the task at hand because i don’t just want my model to predict a yes (1) or No (0)—i want it to predict probabilities that can help me measure how sure it is of its predictions.

```py
def sigmoid(self,Z): return 1/(1+np.exp(-Z))
```




The Loss Function: Next, let’s talk about a neural network’s loss function. The loss function is a way of measuring how good a model’s prediction is so that it can adjust the weights and biases. A loss function must be properly designed so that it can correctly penalize a model that is wrong and reward a model that is right. This means that you want the loss to tell you if a prediction made is far or close to the true prediction. The choice of the loss function is dependent on the task—and for classification problems, you can use cross-entropy loss.

```py
def entropy_loss(self,Y,Yhat):
    nsample=len(Y)
    loss=-1/nsample*(np.sum(np.multiply(np.log(Yhat),Y)+np.multiply(np.log(1-Yhat),(1-Y))))
    return loss
```




Forward Propagation: Now that i have some basic building blocks for my neural network, i’ll move to a very important part of the process called forward propagation. Forward propagation is the name given to the series of computations performed by the neural network before a prediction is made. In my two-layer network, i’ll perform the following computation for forward propagation: Compute the weighted sum between the input and the first layer's weights and then add the bias: Z1 = (W1 * X) + b Pass the result through the ReLU activation function: A1 = Relu(Z1) Compute the weighted sum between the output (A1) of the previous step and the second layer's weights—also add the bias: Z2 = (W2 * A1) + b2 Compute the output function by passing the result through a sigmoid function: A2 = sigmoid(Z2) And finally, compute the loss between the predicted output and the true labels: loss(A2, Y) And there, i have the forward propagation for my two-layer neural network. For a three-layer neural network, i will have to compute Z3 and A2 using W3 and b3 before the output layer.




```py
def forward_propagation(self):
    Z1=self.X.dot(self.params['W1'])+self.params['b1']

    A1=self.relu(Z1)

    Z2=A1.dot(self.params['W2'])+self.params['b2']

    Yhat=self.sigmoid(Z2)

    loss=self.entropy_loss(self.Y,Yhat)

    self.params['Z1']=Z1

    self.params['Z2']=Z2

    self.params['A1']=A1

    return Yhat,loss
```





Backpropagation: Backpropagation is the name given to the process of training a neural network by updating its weights and bias. A neural network learns to predict the correct values by continuously trying different values for the weights and then comparing the losses. If the loss function decreases, then the current weight is better than the previous, or vice versa. This means that the neural net has to go through many training (forward propagation) and update (backpropagation) cycles in order to get the best weights and biases. This cycle is what we generally refer to as the training phase, and the process of searching for the right weights is called optimization. Now the question is, how do you code a neural network to correctly adjust its weights with respect to the loss it calculates. Well, thanks to mathematics, we can use calculus to do this effectively. 




```py
def back_propagation(self,Yhat):
    m=2

    dZ2 = Yhat-self.Y

    dW2 = (1/m)*np.dot(self.params['A1'].T,self.params['Z2'])

    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)


    dZ1 = np.dot(dZ2,self.params['W2'].T)*(1-np.power(self.params['A1'],2))

    dW1 = (1/m)*np.dot(self.X.T,dZ1)

    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

```
    WEIGHT UPDATES WITH CALCULATED GRADIENTS:
```py
    self.params['W1']=self.params['W1']-self.learning_rate*dW1

    self.params['b1']=self.params['b1']-self.learning_rate*db1

    self.params['W2']=self.params['W2']-self.learning_rate*dW2

    self.params['b2']=self.params['b2']-self.learning_rate*db2
```




Optimization and Training of the Neural Network: In the previous section, we used calculus to compute the derivatives of the weights and biases with respect to the loss. The model now knows how to change them. To automatically use this information to update the weights and biases, a neural network must perform hundreds, thousands, and even millions of forward and backward propagations. That is, in the training phase, the neural network must perform the following: Forward propagation Backpropagation Weight updates with calculated gradients Repeat

```py
def fit(self,X,Y):
    self.X=X
    self.Y=Y
    self.init_weight()

    for i in range(self.iterations):
        Yhat,loss=self.forward_propagation()
        self.back_propagation(Yhat)

        self.loss.append(loss)
```




Making Predictions: To make predictions, i simply make a forward pass on the test data. That is, me using the saved weights and biases from the training phase. To make the process easier, i’ll add a function to my NeuralNetwork class called predict:

The function passes the data through the forward propagation layer and computes the prediction using the saved weights and biases. The predictions are probability values ranging from 0 to 1. In order to interpret these probabilities, you can either round up the values or use a threshold function. To keep things simple, we just rounded up the probabilities. Putting It Together

```py
def predict(self,X):
    Z1=X.dot(self.params['W1'])+self.params['b1']
    A1=self.relu(Z1)
    Z2=A1.dot(self.params['W2'])+self.params['b2']
    pred=self.sigmoid(Z2)
    return np.round(pred)
```


