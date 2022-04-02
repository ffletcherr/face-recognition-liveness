from facenet_pytorch import InceptionResnetV1
from glob import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
pil2tensor = transforms.ToTensor()
plt.ioff()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
output_path = 'results'
images_list = sorted(glob('faces/*/*'))
faces  =  []
names = []
for j,image_path in enumerate(images_list): 
    print(image_path)
    names.append(image_path.split('/')[-2]+'-{}'.format(image_path.split('/')[-1].split('.')[0]))
    img = Image.open(image_path).convert('RGB')
    img_cropped = pil2tensor(img)
    faces.append(img_cropped)
    
aligned = torch.stack(faces).to(device)
embeddings = resnet(aligned).detach().cpu()
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
result = pd.DataFrame(embeddings,  index=names).astype('float16')
#%%
th = .7
for name in result.T:
    print(name)
    row = result.loc[name]
    diff = result  -  row
    norm = np.linalg.norm(diff,axis=1)
    rank = norm.argsort()
    min2 = rank[1]
    res_name = result.iloc[min2].name
    disim = norm[min2]
    if disim < th:
        print(min2,disim,res_name)
        print('\n')
        query = Image.open(f'faces/{name[:4]}/{name[5:]}.jpg')
        match = Image.open(f'faces/{res_name[:4]}/{res_name[5:]}.jpg')
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(query)
        ax[1].imshow(match)
        plt.title(str(disim))
        plt.savefig(os.path.join(output_path,name+'.jpg'))
        plt.close()
        

