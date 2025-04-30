from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    layer_f = Logistic()
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, size_list=None, lambda_list=None, kernel_size=[3, 3]):
        self.size_list = size_list
        self.lambda_list = lambda_list
        self.kernel_size = kernel_size

        if size_list is not None:
            self.layers = []
            for i in range(len(size_list) - 3):
                layer = conv2D(size_list[i], size_list[i + 1], kernel_size=self.kernel_size)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                self.layers.append(ReLU())

            flatten_dim = size_list[len(size_list) - 3] * 26 * 26 #! Hard-coding here: 28 -3+1
            flatten_layer = Flatten() # flatten layer
            self.layers.append(flatten_layer)
            # the last two layers are fully connected layers.
            fc1_layer = Linear(in_dim=flatten_dim, out_dim=size_list[-2])
            fc2_layer = Linear(in_dim=size_list[-2], out_dim=size_list[-1])
            if lambda_list is not None:
                fc1_layer.weight_decay = True
                fc1_layer.weight_decay_lambda = lambda_list[-2]
                fc2_layer.weight_decay = True
                fc2_layer.weight_decay_lambda = lambda_list[-1]
            self.layers.append(fc1_layer)
            self.layers.append(ReLU())
            self.layers.append(fc2_layer)
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.lambda_list = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = conv2D(self.size_list[i], self.size_list[i + 1], self.kernel_size)
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                self.layers.append(layer)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.lambda_list]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
    
    
class Model_Bottleneck(Layer):
    """
    A model with conv2D layers and bottleneck layers.
    """
    def __init__(self, size_list=None, lambda_list=None, kernel_size=3):
        assert len(size_list) == 5, 'The size_list should be 5.'
        self.size_list = size_list
        self.lambda_list = lambda_list
        self.kernel_size = kernel_size
        
        if size_list is not None:
            self.layers = []
            # the first layer is a conv2D layer.
            layer = conv2D(size_list[0], size_list[1], kernel_size=3)
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[0]
            self.layers.append(layer)

            # the second layer is a bottleneck layer.
            layer = Bottleneck(size_list[1], size_list[2], kernel_size=3)
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[1]
            self.layers.append(*layer.sublayers)

            flatten_layer = Flatten() # flatten layer
            self.layers.append(flatten_layer)
            # the last two layers are fully connected layers.
            fc1_layer = Linear(in_dim=size_list[-3], out_dim=size_list[-2])
            fc2_layer = Linear(in_dim=size_list[-2], out_dim=size_list[-1])
            if lambda_list is not None:
                fc1_layer.weight_decay = True
                fc1_layer.weight_decay_lambda = lambda_list[-2]
                fc2_layer.weight_decay = True
                fc2_layer.weight_decay_lambda = lambda_list[-1]
            self.layers.append(fc1_layer)
            self.layers.append(fc2_layer)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.lambda_list = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = conv2D(self.size_list[i], self.size_list[i + 1], self.kernel_size)
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                self.layers.append(layer)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.lambda_list]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_MLP_Dropout(Layer):
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_prob=0.3):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_prob = dropout_prob

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(
                    in_dim=size_list[i],
                    out_dim=size_list[i + 1],
                    weight_decay=True if lambda_list is not None else False,
                    weight_decay_lambda=lambda_list[i] if lambda_list is not None else 0
                )
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        self.layers.append(Logistic())
                    elif act_func == 'ReLU':
                        self.layers.append(ReLU())
                    self.layers.append(Dropout(p=self.dropout_prob))  # Dropout

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads