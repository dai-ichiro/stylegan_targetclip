import os

import numpy as np
import torch
import torchvision

from models.stylegan2.model import Generator
import math
import copy

checkpoint = 'stylegan2-ffhq-config-f.pt'

g_ema = Generator(1024, 512, 8)
g_ema.load_state_dict(torch.load(checkpoint)["g_ema"], strict=False)
g_ema.eval().to('cuda')

dirs = {
  'Elsa': 'dirs/elsa.npy',
  'Pocahontas': 'dirs/pocahontas.npy',
  'Keanu Reeves': 'dirs/keanu.npy',
  'Trump': 'dirs/trump.npy',
  'Joker': 'dirs/joker.npy',
  'Ariel': 'dirs/ariel.npy',
  'Doc Brown': 'dirs/doc.npy',
  'Beyonce': 'dirs/beyonce.npy',
  'Morgan Freeman': 'dirs/morgan.npy',
}

target = 'Elsa' 

source = torch.from_numpy(np.load('test_result.npy'))
source = source.unsqueeze(0)
dir = torch.from_numpy(np.load(dirs[target]))

os.makedirs('result', exist_ok = True)
for alpha in [x / 10 for x in range(0, 21)]:
    
    source_amp, _ = g_ema([(source + dir * alpha).to('cuda')], 
                            input_is_latent=True,
                            randomize_noise=False)

    torchvision.utils.save_image(source_amp, "result/results_manipulated_%f.png"%alpha, normalize=True, range=(-1, 1))