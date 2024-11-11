import torch.nn as nn

class Hook():
    def __init__(self,
                 model,
                 layer_names: list):
        """init the hook. Only support forward hooks currently.

        Args:
            model(nn.model): the model to fix on
            layer_names(list): the list of layer names(in pytorch) to fix on,written as path
        
        Returns:
            None
        """
        self.model = model
        self.forward_out_hooks = {}
        self.handles_forward = {}
        self.layer_names = layer_names
        self.register_hooks()

    def __del__(self) -> None:
        """delete the hook
        Args:
            None

        Returns:
            None
        """
        self.remove_hooks()

    def _get_layer(self,
                   layer_name: str) -> nn.Module:
        """get layer module with layer name

        Args:
            layer_name(str): the name of the layer(in pytorch)

        Returns:
            pre_module(nn.Module): the layer module
        """
        layer_list = layer_name.split("/")
        prev_module = self.model
        for layer in layer_list:
            prev_module = prev_module._modules[layer]

        return prev_module

    def _register_single_hook(self,
                              layer_name: str) -> None:
        """register hook on a single layer

        Args:
            layer_name(str): the name of the layer to be registered hook on
        
        Returns:
            None
        """
        # the forward hook function,record the features
        def forward_hook_fn(module, input, output):
            self.forward_out_hooks[layer_name] = output

        layer = self._get_layer(layer_name)
        handle_forward = layer.register_forward_hook(forward_hook_fn)
        self.handles_forward[layer_name] = handle_forward

    def _remove_single_hook(self,
                            layer_name: str) -> None:
        """remove hoook from a single layer
        
        Args:
            layer_name(str): the name of the layer to be removed hook from

        Returns:
            None
        """
        handle_forward = self.handles_forward[layer_name]
        handle_forward.remove()

    def register_hooks(self) -> None:
        """register hooks on all layers
        Args:
            None

        Returns:
            None
        """
        for layer_name in self.layer_names:
            self._register_single_hook(layer_name)

    def remove_hooks(self) -> None:
        """remove hooks from all layers
        Args:
            None

        Returns:
            None
        """
        for layer_name in self.layer_names:
            self._remove_single_hook(layer_name)

    def get_features(self, layer_name) -> dict:
        """get features from all registered layers,can only be called after a forward propagation
        Args:
            None

        Returns:
            features_dict(dict): the dict contains all features out from defferent layers,the keys are the layers' name
        """
        return self.forward_out_hooks[layer_name]