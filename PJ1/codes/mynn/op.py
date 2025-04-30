from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}
        self.optimizable = True

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
    
    def sync_params(self):
        self.W = self.params['W']
        self.b = self.params['b']
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = X @ self.W + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input.shape[0] == grad.shape[0], "The input previous grad and X should have the same batch size."
        assert self.input.shape[1] == self.W.shape[0], "The input previous grad and W should have the same dimension."

        self.grads['W'] = self.input.T @ grad / self.input.shape[0] # batch normalization of gradients
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / self.input.shape[0]
        if self.weight_decay: # or implemented in L2Regularization
            self.grads['W'] += self.weight_decay_lambda * self.W

        output = grad @ self.W.T
        return output
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W' : None, 'b' : None}
        self.input = None
        
        self.params = {'W' : self.W, 'b' : self.b}
        self.optimizable = True
        
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.k_H = kernel_size[0]
        self.k_W = kernel_size[1]
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def sync_params(self):
        self.W = self.params['W']
        self.b = self.params['b']

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape
        assert in_channels == self.W.shape[1], "The input channels and W channels should be the same."

        new_H = (H - self.k_H) // self.stride + 1 #! no padding
        new_W = (W - self.k_W) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, new_H, new_W))

        for i in range(new_H):
            for j in range(new_W):
                output[:, :, i, j] = np.tensordot(X[:, :, i*self.stride : i*self.stride+self.k_H, j*self.stride : j*self.stride+self.k_W], self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b[0, :, 0, 0] #! no padding
        
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        # assert ...
        batch_size, in_channels, H, W = self.input.shape
        _, out_channels, new_H, new_W = grads.shape
        
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.input)

        for i in range(new_H):
            for j in range(new_W):
                input_slice = self.input[:, :, i*self.stride : i*self.stride+self.k_H, j*self.stride : j*self.stride+self.k_W]

                dW += np.tensordot(grads[:, :, i, j], input_slice, axes=([0], [0]))

                db += np.sum(grads[:, :, i, j], axis=0).reshape(self.b.shape)

                dX[:, :, i*self.stride : i*self.stride+self.k_H, j*self.stride : j*self.stride+self.k_W] += np.tensordot(grads[:, :, i, j], self.W, axes=([1], [0]))
        dW /= batch_size
        db /= batch_size
        dX /= batch_size
        
        # Apply L2 regularization to dW if enabled
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}  
        
class Flatten(Layer):
    """
    A flatten layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = X.reshape(X.shape[0], -1)
        return output
    
    def backward(self, grads):
        assert self.input.shape[0] == grads.shape[0], "The input previous grad and X should have the same batch size."
        assert np.prod(self.input.shape[1:]) == np.prod(grads.shape[1:]), "The input previous grad and X should have the same dimension."
        output = grads.reshape(self.input.shape)
        return output
    
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output
    
class Logistic(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = 1 / (1 + np.exp(-X))
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = grads * (1 - self.input) * self.input
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.grads = None
        self.eps = 1e-10 # to avoid log(0)

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        assert predicts.shape[0] == labels.shape[0], "The batch size of predicts and labels should be the same."
        assert predicts.shape[1] == self.max_classes, "The number of classes should be the same."
        self.grads = np.zeros_like(predicts)
        if self.has_softmax:
            predicts = softmax(predicts)
        self.predicts = predicts
        self.labels = labels

        selected_probs = predicts[np.arange(predicts.shape[0]), labels]
        loss = np.sum(-np.log(np.clip(selected_probs, self.eps, 1.0))) / predicts.shape[0]
        
        return loss
    
    def backward(self):
        """
        This function generates the gradients for the previous layer.
        """
        batch_size = self.predicts.shape[0]
        
        if self.has_softmax:
            # Softmax + CrossEntropy: grads = (p - one_hot(labels)) / batch_size
            one_hot_labels = np.zeros_like(self.grads)
            one_hot_labels[np.arange(batch_size), self.labels] = 1
            self.grads = (self.predicts - one_hot_labels) / batch_size
        else:
            # CrossEntropy only: grads = (-1/p[labels]) / batch_size for correct class, 0 otherwise
            self.grads.fill(0)
            self.grads[np.arange(batch_size), self.labels] = -1.0 / np.clip(self.predicts[np.arange(batch_size), self.labels], self.eps, 1.0)
            self.grads /= batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear. It wraps the loss function.
    """
    def __init__(self, model=None, lambda_=1e-8, loss_fn=None) -> None:
        super().__init__()
        self.model = model
        self.lambda_ = lambda_
        self.loss_fn = loss_fn
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        loss = self.loss_fn(predicts, labels)
        for layer in self.model.layers:
            if hasattr(layer, 'weight_decay'):
                if layer.weight_decay:
                    loss += np.sum(layer.W ** 2) * self.lambda_ / 2.0
        return loss
    
    def backward(self):
        self.loss_fn.backward()
        for layer in self.model.layers:
            if hasattr(layer, 'weight_decay'):
                if layer.weight_decay:
                    layer.grads['W'] += layer.W * self.lambda_

class Dropout(Layer):
    """
    A dropout layer.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.mask = None
        
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.mask = np.random.binomial(1, 1-self.p, size=X.shape) / (1-self.p)
        return X * self.mask
    
    def backward(self, grads):
        return grads * self.mask
    
class Bottleneck(Layer):
    """
    A bottleneck layer with channels shrunk to one-fourth of the original size.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.conv1 = conv2D(in_channels, out_channels//4, kernel_size=1, stride=stride, padding=padding, initialize_method=initialize_method, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda)
        self.conv2 = conv2D(out_channels//4, out_channels//4, kernel_size=kernel_size, stride=stride, padding=padding, initialize_method=initialize_method, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda)
        self.conv3 = conv2D(out_channels//4, out_channels, kernel_size=1, stride=stride, padding=padding, initialize_method=initialize_method, weight_decay=weight_decay, weight_decay_lambda=weight_decay_lambda)
        
        self.sublayers = [self.conv1, self.conv2, self.conv3]
        self.params = {'conv1': self.conv1.params, 'conv2': self.conv2.params, 'conv3': self.conv3.params}
        self.optimizable = True

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        return output
    
    def backward(self, grads):
        dX = self.conv3.backward(grads)
        dX = self.conv2.backward(dX)
        dX = self.conv1.backward(dX)
        return dX
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition