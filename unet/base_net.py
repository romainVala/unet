import tarfile
import tempfile
import json
import shutil

import torch
import torch.nn as nn


class BaseNet(nn.Module, object):

    def __init__(self, **kwargs):
        super(BaseNet, self).__init__()

        # Keep all the __init__parameters for saving/loading
        self.net_parameters = kwargs

    def save(self, filename):
        with tarfile.open(filename, "w") as tar:
            temporary_directory = tempfile.mkdtemp()
            name = "{}/net_params.json".format(temporary_directory)
            json.dump(self.net_parameters, open(name, "w"))
            tar.add(name, arcname="net_params.json")
            name = "{}/state.torch".format(temporary_directory)
            torch.save(self.state_dict(), name)
            tar.add(name, arcname="state.torch")
            shutil.rmtree(temporary_directory)
        return filename

    @classmethod
    def load(cls, filename, device=torch.device('cpu')):
        with tarfile.open(filename, "r") as tar:
            net_parameters = json.loads(
                tar.extractfile("net_params.json").read().decode("utf-8"))
            path = tempfile.mkdtemp()
            tar.extract("state.torch", path=path)
            net = cls(**net_parameters)
            net.load_state_dict(
                torch.load(
                    path + "/state.torch",
                    map_location=device,
                )
            )
        return net, net_parameters


def clean_locals(**kwargs):
    del kwargs['self']
    del kwargs['__class__']
    return kwargs
