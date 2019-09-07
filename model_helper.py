import os, glob
import torch

def load_checkpoint(model, checkpoint_path):
    checkpoint_path = glob.glob(f'results/{checkpoint_path}*')[0]
    checkpoint_model = torch.load(checkpoint_path)

    model_dict = model.state_dict()
    # 1.filter out unnecessary keys
    checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(checkpoint_model)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print('=========================================')
    print(f'checkpoint loaded from {checkpoint_path}')
    print('=========================================')

    return model
