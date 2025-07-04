# DINOv2+SEM

**Authors of DINOv2**:
Maxime Oquab,
Timothée Darcet,
Théo Moutakanni,
Huy V. Vo,
Marc Szafraniec,
Vasil Khalidov,
Patrick Labatut,
Armand Joulin,
Piotr Bojanowski

**Authors of SEMs modifications**:
Samuel Lavoie,
Michael Noukhovitch,
Aaron Courville

[[`SEM paper`](https://arxiv.org/abs/2204.00616)] [`DLC paper`](TODO)] [`Dinov2 Paper #1`](https://arxiv.org/abs/2304.07193)] [`Dinov2 Paper #2`](https://arxiv.org/abs/2309.16588)] [[`BibTeX`](#citing-dinov2)]

PyTorch implementation for DINOv2 + SEM. Training on 4 H100.

### Training Dinov2 + SEM

Update the config file `vitl14_sem_L512.yaml` to specify the ROOT_PATH where the base dinov2 model and the extra datasets is save. The extra
dataset can be generated using the procedure from [dinov2's repos](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#data-preparation).
After the preparation, the SEM encoder can be trained as follows. You might have to update `dinov2/run/submit.py` depending on your SLURM setup.
```shell
python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/train/vitl14_sem_L512.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

The training code saves the weights of the teacher in the `eval` folder every 6250 iterations for evaluation.

Training time is approximately 1.2 days.

## Evaluation

The training code regularly saves the teacher weights. In order to evaluate the model, run the following evaluation on a single node:

### Linear classification with data augmentation on ImageNet-1k

```shell
python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitl14_L512.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/linear \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

## Citing DINOv2 + SEM

If you find this repository useful, please consider giving a star :star: and citations:

```
@inproceedings{
  lavoie2023simplicial,
  title={Simplicial Embeddings in Self-Supervised Learning and Downstream Classification},
  author={Samuel Lavoie and Christos Tsirigotis and Max Schwarzer and Ankit Vani and Michael Noukhovitch and Kenji Kawaguchi and Aaron Courville},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=RWtGreRpovS}
}
```

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

```
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```
