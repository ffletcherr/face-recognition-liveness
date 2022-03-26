"""
Created on Tue Jul 27 10:19:52 2021

@author: hm
"""

import numpy as np

import torch
from torch import nn
from torchvision import models

from time import time 
from facenet_pytorch import MTCNN, InceptionResnetV1

from torch.autograd import Variable
import torchvision.transforms as transforms
import logging

class faceMTCNN():

    def __init__(self,period=5):
        self.period = period
        self.time   = time()
        self.buff   = False
        self.mtcnn  = MTCNN()
        
    def schedule(self,img):
        if time() - self.time > self.period:
            self.buff = True
            crop, boxes = self.mtcnn(img)
            self.time   = time()
            return crop, boxes
        else:
            self.buff = False
            return None,None
    
    def detect(self,img):
        crop, boxes = self.mtcnn(img)
        return crop, boxes
        
        
class embedResnet():
    def __init__(self,period=15):
        self.period = period
        self.time   = time()
        self.last_face = None
        self.resnet  = InceptionResnetV1(pretrained='vggface2').eval()
        
    def schedule(self):
        if time() - self.time > self.period:
            crop = self.last_face
            if crop is not None:
                embeddings = self.resnet(crop.unsqueeze(0)).detach().cpu().numpy()
                self.time   = time()
                return embeddings
            else:
                return None
        else:
            return None
        
    def verify(self,crop):
        embeddings = self.resnet(crop.unsqueeze(0)).detach().cpu().numpy()
        return embeddings




class DeepPixBiS(nn.Module):
    """ The class defining Deep Pixelwise Binary Supervision for Face Presentation
    Attack Detection:

    Reference: Anjith George and Sébastien Marcel. "Deep Pixel-wise Binary Supervision for 
    Face Presentation Attack Detection." In 2019 International Conference on Biometrics (ICB).IEEE, 2019.

    Attributes
    ----------
    pretrained: bool
        If set to `True` uses the pretrained DenseNet model as the base. If set to `False`, the network
        will be trained from scratch. 
        default: True      
    """

    def __init__(self, pretrained=True):

        """ Init function

        Parameters
        ----------
        pretrained: bool
            If set to `True` uses the pretrained densenet model as the base. Else, it uses the default network
            default: True
        """
        super(DeepPixBiS, self).__init__()

        dense = models.densenet161(pretrained=pretrained)

        features = list(dense.features.children())

        self.enc = nn.Sequential(*features[0:8])

        self.dec=nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.linear=nn.Linear(14*14,1)


    def forward(self, x):
        """ Propagate data through the network

        Parameters
        ----------
        img: :py:class:`torch.Tensor` 
          The data to forward through the network. Expects RGB image of size 3x224x224

        Returns
        -------
        dec: :py:class:`torch.Tensor` 
            Binary map of size 1x14x14
        op: :py:class:`torch.Tensor`
            Final binary score.  

        """
        enc = self.enc(x)

        dec=self.dec(enc)

        dec=nn.Sigmoid()(dec)

        dec_flat=dec.view(-1,14*14)

        op=self.linear(dec_flat)

        op=nn.Sigmoid()(op)
 
        return dec,op
    
    



logger = logging.getLogger("bob.paper.deep_pix_bis_pad.icb2019")
class DeepPixBiSExtractor():
  """ The class implementing the FASNet score computation.

  Attributes
  ----------
  network: :py:class:`torch.nn.Module`
      The network architecture
  transforms: :py:mod:`torchvision.transforms`
      The transform from numpy.array to torch.Tensor

  """
  
  def __init__(self, transforms = transforms.Compose(
                                    [transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])]),
                model_file=None, scoring_method='pixel_mean'):
          
    """ Init method

    Parameters
    ----------
    model_file: str
      The path of the trained PAD network to load
    transforms: :py:mod:`torchvision.transforms` 
      Tranform to be applied on the image
    scoring_method: str
      The scoring method to be used to get the final score, 
      available methods are ['pixel_mean','binary','combined']. 
    
    """
    
    # model
    self.transforms = transforms 
    self.network = DeepPixBiS(pretrained=True)
    self.scoring_method = scoring_method
    self.available_scoring_methods=['pixel_mean','binary','combined']

    logger.debug('Scoring method is : {}'.format(self.scoring_method.upper())) 

    if model_file is None:
      # do nothing (used mainly for unit testing) 
      logger.debug("No pretrained file provided")
      pass
    else:


      # With the new training
      logger.debug('Starting to load the pretrained PAD model')

      try:
        cp = torch.load(model_file)
      except:
        try:
          cp= torch.load(model_file,map_location=lambda storage,loc:storage) 
        except:
          raise ValueError('Could not load the model')

      if 'state_dict' in cp:
        self.network.load_state_dict(cp['state_dict'])
      else: ## check this part
        self.network.load_state_dict(cp)

      logger.debug('Loaded the pretrained PAD model')    
 
    self.network.eval()

  def __call__(self, image):
    """ Extract features from an image

    Parameters
    ----------
    image : 3D :py:class:`numpy.ndarray`
      The image to extract the score from. Its size must be 3x224x224;
      
    Returns
    -------
    output : float
      The extracted feature is a scalar values ~1 for bonafide and ~0 for PAs
    
    """
   
    input_image = np.rollaxis(np.rollaxis(image, 2),2) # changes to 128x128xnum_channels
    input_image = self.transforms(input_image)
    input_image = input_image.unsqueeze(0)
    
    output = self.network.forward(Variable(input_image))
    output_pixel = output[0].data.numpy().flatten()
    output_binary = output[1].data.numpy().flatten()

    if self.scoring_method=='pixel_mean':
      score=np.mean(output_pixel)
    elif self.scoring_method=='binary':
      score=np.mean(output_binary)
    elif self.scoring_method=='combined':
      score= (np.mean(output_pixel)+np.mean(output_binary))/2.0
    else:
      raise ValueError('Scoring method {} is not implemented.'.format(self.scoring_method))

    # output is a scalar score

    return score
