from core.models.segnet import SegNet
from core.models.unet import UNet
from core.models.unet3d import UNet3D
from core.models.deconvnet import DeconvNet


def load_empty_model(architecture, n_classes):
    models = {
        'segnet': SegNet,
        'unet': UNet,
        'unet3d': UNet3D,
        'deconvnet': DeconvNet
    }

    ModelClass = models[architecture]
    model = ModelClass(in_channels=1, num_classes=n_classes)

    return model
