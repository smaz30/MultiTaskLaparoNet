# MultiTaskLaparoNet
<p align="center">
  <h2 align="center">Surgical Instrument Segmentation and Self-Supervised Monocular Depth Estimation in Minimally Invasive Surgery: A Multi-task Learning Approach</h2>
  <h3 align="center">AIME 2025</h3>  
  <h3 align="center"><a href="https://link.springer.com/chapter/10.1007/978-3-031-95838-0_28">Paper</a></h3>
  <div align="center"></div>
  <p align="center">
    <a><strong>Stefano Mazzocchetti</strong></a><sup>1</sup>&nbsp;&nbsp;&nbsp;
    <br />
    <sup>1</sup><strong>eDIMES Lab – Laboratory of Bioengineering, Department of Medical and Surgical Sciences, University of Bologna</strong>    
  </p>
</p>


<!-- Official Implementation of :  _Surgical Instrument Segmentation and Self-Supervised Monocular Depth Estimation in Minimally Invasive Surgery: A Multi-task Learning Approach_,  Accepted at AIME 2025. -->

---
![](https://github.com/smaz30/MultiTaskNet/blob/main/assets/asset_multitask.gif)
---
## Basic Usage
```python
import torch
from net.multitasknet import MultitaskNet

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
state_dict = torch.load("./pretrained/dispnet__convnext_v2_nano_multitask_net.pth.tar")['state_dict']
model = MultitaskNet(convnext_size = 'nano').to(device)
model.load_state_dict(state_dict)
model.eval()
# NOTE: images should be normalized as in ImageNet
dummy_input = torch.randn([1,3,256,320]).to(device)

disparity, mask = model(dummy_input)
```

## Dataset

### SCARED
Please follow [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) to prepare the SCARED dataset.

### Hamlyn
Please download the rectified version of the [Hamlyn Dataset](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion)


## ✏️ 📄 Citation

If you find our work useful in your research please consider citing our paper:
```
@inproceedings{mazzocchetti2025surgical,
  title={Surgical Instrument Segmentation and Self-Supervised Monocular Depth Estimation in Minimally Invasive Surgery: A Multi-task Learning Approach},
  author={Mazzocchetti, Stefano and Cercenelli, Laura and Marcelli, Emanuela},
  booktitle={International Conference on Artificial Intelligence in Medicine},
  pages={283--292},
  year={2025},
  organization={Springer}
}
```
