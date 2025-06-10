from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=<ROOT>, extra=<EXTRA>)
    dataset.dump_extra()
