import torch
from facenet_models import FacenetModel

if False: #torch.backends.mps.is_available():
    device = 'mps'
    print(f'Using mps device')
else:
    device = 'cpu'

model = FacenetModel(device)
