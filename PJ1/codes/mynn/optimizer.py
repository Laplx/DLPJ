from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]
            if hasattr(layer, 'sync_params'):
                    layer.sync_params()


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = {}

        # Avoid same names of params in different layers
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    unique_key = f"layer{layer_idx}_{key}"
                    self.v[unique_key] = np.zeros_like(layer.params[key])

    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    unique_key = f"layer{layer_idx}_{key}"
                    self.v[unique_key] = self.mu * self.v[unique_key] - self.init_lr * layer.grads[key]
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] + self.v[unique_key]
                if hasattr(layer, 'sync_params'):
                    layer.sync_params()