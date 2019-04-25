# AND: Anchor Neighbourhood Discovery

*Published on ICML 2019.*

Pytorch implementation of [Unsupervised Deep Learning by Neighbourhood Discovery](). 

## Highlight
+ We propose the idea of exploiting local neighbourhoods for unsupervised deep learning. This strategy preserves the capability of clustering for class boundary inference whilst minimising the negative impact of class inconsistency typically encountered in clusters.
+ We formulate an *Anchor Neighbouhood Discovery (AND)* approach to progressive unsupervised deep learning. The AND model not only generalises the idea of sample specificity learning, but also additionally considers the originally missing sample-to-sample correlation during model learning by a novel neighbourhood supervision design.
+ We further introduce a curriculum learning algorithm to gradually perform neighbourhood discovery for maximising the class consistency of neighbourhoods therefore enchancing the unsupervised learning capability.

## Reproduction

### Requirements
Python 2.7 and Pytorch 1.0.1 are required. Please refer to `/path/to/AND/requirements.yaml` for other necessary modules or rebuild the conda environment we used for our experiments according to it.

### Usages

1. Clone this repo: `git clone https://github.com/Raymond-sci/AND.git`
2. Download dataset and store them in `/path/to/AND/data`. (Soft link is recommended to avoid redundant copies of datasets)
2. To reproduce our reported result of ResNet18 on CIFAR10, please use the following command:`python main.py --cfgs configs/base.yaml configs/cifar10.yaml`
3. Running on GPUs: code will be ran on CPU by default, use this flag to specify the gpu devices which you want to use
4. To evaluate a trained model, use `--resume` to set the path to the saved checkpoint file and use `--test-only` flag to exit the program after evaluation

Everytime the `main.py` is ran, a new session will be started with the name of current timestamp and all the related files will be stored in folder `sessions/timestamp/` including checkpoints, logs, etc.

More useful options are provided like using various data augmentations, optimizers, learning rate schedules, logging by tfboard, etc. All this features can be utilised by combination of options. Use `python main.py -h` to explore.  
Notice that, we have our own strategy to process configuration which is self-adapting. It means that some arguments will only be triggered to show when pre-condition is met, e.g. if learning rate schedule is set to `fixed`, only the argument `--base-lr` is required. However, if it is set to `step`, a few more value will be requested like `--lr-decay-offset`, `--lr-decay-rate`, etc.

### Pre-trained model
Coming soon...

## License
This project is licensed under the MIT License. You may find out more [here](LICENSE).

## Reference
If you use this code, please cite the following paper:

Jiabo Huang, Qi Dong, Shaogang Gong and Xiatian Zhu. "Unsupervised Deep Learning by Neighbourhood Discovery." Proc. ICML (2019).

```
@InProceedings{huang2018and,
  title={Unsupervised Deep Learning by Neighbourhood Discovery},
  author={Jiabo Huang, Qi Dong, Shaogang Gong and Xiatian Zhu},
  booktitle={Proceedings of the International Conference on machine learning (ICML)},
  year={2019},
}
```