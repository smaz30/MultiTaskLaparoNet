# MultiTaskLaparoNet
Official Implementation of :  _Surgical Instrument Segmentation and Self-Supervised Monocular Depth Estimation in Minimally Invasive Surgery: A Multi-task Learning Approach_,  Accepted at AIME 2025.

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
dummy_input = torch.randn([1,3,256,320]).to(device)

disparity, mask = model(dummy_input)
```

## Dataset

### SCARED
Please follow [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) to prepare the SCARED dataset.

### Hamlyn
Please download the rectified version of the [Hamlyn Dataset](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion)