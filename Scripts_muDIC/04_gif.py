from PIL import Image
import glob
import os

# Create the frames
frames = []
folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_8\1-14442C107152C0D200\left\prueba\00_Results'
imgs = glob.glob(os.path.join(folder,"*.png"))

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save(os.path.join(folder,'2.8.gif'), format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)
