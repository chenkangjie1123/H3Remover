import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device


class Storage:
    def __init__(self, module):
        if any(p.device.type != 'cpu' or p.requires_grad for p in module.parameters()):
            raise ValueError('Offload requires frozen CPU parameters.')
        self.module = module
        self.parameters = {n: p.detach() for n, p in module.named_parameters()}
        self.buffers = {n: b.detach() for n, b in module.named_buffers()}

    def restore(self):
        with torch.no_grad():
            for name, tensor in self.parameters.items():
                self.module.get_parameter(name).data = tensor
            for name, tensor in self.buffers.items():
                owner, key = name.rsplit('.', 1) if '.' in name else ('', name)
                self.module.get_submodule(owner)._buffers[key] = tensor


class ComponentOffload:
    def __init__(self, components, device):
        self.device = torch.device(device)
        self.current = None
        self.storage = {name: Storage(module) for name, module in components.items()
                        if isinstance(module, torch.nn.Module)}
        manager = self

        class Hook(ModelHook):
            no_grad = True

            def __init__(self, name):
                self.name = name
                self.execution_device = manager.device

            def pre_forward(self, module, *args, **kwargs):
                manager.onload(self.name)
                return send_to_device(args, manager.device), send_to_device(kwargs, manager.device)

        for name, storage in self.storage.items():
            add_hook_to_module(storage.module, Hook(name))

    def onload(self, name):
        if self.current == name:
            return
        if self.current is not None:
            self.storage[self.current].restore()
            torch.cuda.empty_cache()
        self.storage[name].module.to(self.device)
        self.current = name

    def close(self):
        for storage in self.storage.values():
            storage.restore()
        self.current = None
        torch.cuda.empty_cache()
