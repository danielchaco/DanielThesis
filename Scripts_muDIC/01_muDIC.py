import muDIC as dic
import os
import matplotlib.pyplot as plt

rot_cropp_imgs = 'C:\\Users\\dmchacon\\Documents\\OAK-D Videos\\2_8\\1-14442C107152C0D200\\left\\prueba\\rot_cropped'
destination_folder = os.path.join(os.path.split(rot_cropp_imgs)[0],'00_Results')
os.makedirs(destination_folder,exist_ok=True)

image_stack = dic.image_stack_from_folder(rot_cropp_imgs,file_type=".png")
mesher = dic.Mesher()
mesh = mesher.mesh(image_stack)
inputs = dic.DICInput(mesh,image_stack)
dic_job = dic.DICAnalysis(inputs)
results = dic_job.run()
fields = dic.Fields(results)
true_strain = fields.true_strain()
viz = dic.Visualizer(fields,images=image_stack)

for i in range(len(image_stack)):
    fig = viz.show(field="True strain", component = (0,1), frame = i)
    fig.savefig(os.path.join(destination_folder,str(i)+'.png'))