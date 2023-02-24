import glob
import muDIC as dic
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from natsort import natsorted

def find_file_names(path, type=".png"):
    return natsorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith(type)])


# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_8\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_8\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_8\1-14442C107152C0D200\left_0\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_1\4-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_1\4-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_2\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_2\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_4\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_4\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_5\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_5\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_11\1-14442C10E17CC0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_11\1-14442C10E17CC0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_14\5-14442C10E17CC0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_14\5-14442C10E17CC0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C10E17CC0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C10E17CC0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_17\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_17\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_19\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_19\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_2\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_2\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_4\1-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_4\1-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_6\2-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_6\2-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_9\4-14442C107152C0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_9\4-14442C107152C0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_11\1-14442C10E17CC0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_11\1-14442C10E17CC0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_12\1-14442C10E17CC0D200\left\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_12\1-14442C10E17CC0D200\right\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_14\1-14442C10E17CC0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_14\1-14442C107152C0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_16\1-14442C10E17CC0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_16\1-14442C107152C0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_18\1-14442C10E17CC0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_18\1-14442C107152C0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_20\1-14442C10E17CC0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_20\1-14442C107152C0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C10E17CC0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C107152C0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_7\2-14442C10E17CC0D200\color\cropped'
# rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_7\2-14442C107152C0D200\color\cropped'
rot_cropp_imgs = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_1\right\cropped'

destination_folder = os.path.split(rot_cropp_imgs)[0]
ID = destination_folder.split('\\')[5]
cam = 'left' if 'left' in destination_folder else 'right'
out_path = os.path.join(destination_folder,f'{ID}_DIC_{cam}.csv')
os.makedirs(destination_folder,exist_ok=True)

# image_paths = glob.glob(os.path.join(rot_cropp_imgs,'*.png'))
image_paths = find_file_names(rot_cropp_imgs,type=".png")
image_stack = dic.image_stack_from_folder(rot_cropp_imgs,file_type=".png")
mesher = dic.Mesher()
mesh = mesher.mesh(image_stack)
inputs = dic.DICInput(mesh,image_stack)
dic_job = dic.DICAnalysis(inputs)
results = dic_job.run()
fields = dic.Fields(results)
true_strain = fields.true_strain()

strain = []
for i in tqdm(range(true_strain.shape[-1])):
    E = true_strain[0][:,:,:,:,i]
    shear_strain = E[0,1]
    strain.append([float(os.path.basename(image_paths[i]).split('.')[0]),np.min(shear_strain),np.max(shear_strain),np.mean(shear_strain),np.median(shear_strain)])

df = pd.DataFrame(strain,columns=['microSeconds','min','max','mean','median'])
df.sort_values('microSeconds',ignore_index=True,inplace=True)
df.to_csv(out_path,index=False)