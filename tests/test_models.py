import torch
from src.models import build_model


def test_all_model_dataset_shapes():
    for image_size, classes in [(32, 10), (32, 100), (64, 200)]:
        for model_name in ["resnet18", "mobilenet_v2"]:
            model = build_model(model_name, classes, "group", image_size)
            output = model(torch.randn(2, 3, image_size, image_size))
            assert output.shape == (2, classes)
