"""Common torchvision transforms used across training and evaluation."""
from torchvision import transforms


class ResizeImage():
    """Callable resize helper compatible with torchvision transforms pipelines."""

    def __init__(self, size):
        """Store the desired output size as a tuple."""
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """Return the resized PIL image."""
        th, tw = self.size
        return img.resize((th, tw))


def rr_train(resize_size=(256, 256)):
    """Return the default train-time transformation pipeline."""
    return transforms.Compose([
        transforms.Scale(resize_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def rr_eval(resize_size=(256, 256)):
    """Return the default eval-time transformation pipeline."""
    return transforms.Compose([
        transforms.Scale(resize_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
