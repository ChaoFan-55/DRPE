# 数据放置

请在项目根目录下保持：

```text
data/
├── cifar-10-batches-py/      # torchvision也可自动下载
├── cifar-100-python/         # torchvision也可自动下载
└── tiny-imagenet-200/
    ├── train/
    ├── val/
    ├── wnids.txt
    └── words.txt
```

CIFAR-10/100配置使用 `data_root=./data`。Tiny-ImageNet不会自动下载，缺失时程序会给出明确报错。
