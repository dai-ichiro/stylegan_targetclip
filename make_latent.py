from argparse import Namespace
import numpy as np

import torch
from torchvision import transforms

from models.psp import pSp  # we use the pSp framework to load the e4e encoder.

model_path = 'e4e_ffhq_encode.pt'
ckpt = torch.load(model_path)

opts = ckpt['opts']
opts['checkpoint_path'] = model_path

opts= Namespace(**opts)
net = pSp(opts)
net.eval().to('cuda')

from PIL import Image
'''
image_path = 'test.jpg'
original_image = Image.open(image_path)
original_image = original_image.convert("RGB")
'''

import dlib
from utils.alignment import align_face

def run_alignment(image_path):
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 

image_path = 'test.jpg'
input_image = run_alignment(image_path)

transform_fn = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

transformed_image = transform_fn(input_image)

with torch.no_grad():
    images, latents = net(transformed_image.unsqueeze(0).to('cuda').float(), randomize_noise=False, return_latents=True)
    #images, latents = run_on_batch(transformed_image.unsqueeze(0), net)

result_image, latent = images[0], latents[0]

result_array = latent.to('cpu').numpy()
np.save('test_result', result_array)