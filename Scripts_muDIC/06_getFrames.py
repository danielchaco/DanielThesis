from datetime import datetime, timezone
import av
import av.datasets
import glob
import os

import pandas as pd
from tqdm import tqdm

def get_frames(vpath,out_folder = None):
    content = av.datasets.curated(vpath)
    if out_folder:
        out_folder = os.path.join(out_folder,'\\'.join(vpath.split('.mp4')[0].split('\\')[5:]))
    else:
        out_folder = vpath.split('.mp4')[0]
    os.makedirs(out_folder, exist_ok=True)

    with av.open(content) as container:
        # Signal that we only want to look at keyframes.
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            # We use `frame.pts` as `frame.index` won't make must sense with the `skip_frame`.
            frame.to_image().save(
                os.path.join(out_folder, "{:04d}.png".format(frame.pts)),
                quality=80,
            )

folder_list = [
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\2_14\5-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\2_17\1-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\2_19\1-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_2\1-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_4\1-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_6\2-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_7\2-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_7\2-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_9\4-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_11\1-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_12\1-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_14\1-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_14\1-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_16\1-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_16\1-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_18\1-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_18\1-14442C107152C0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_20\1-14442C10E17CC0D200',
    # r'C:\Users\dmchacon\Documents\OAK-D Videos\3_20\1-14442C107152C0D200',
    r'C:\Users\dmchacon\Documents\OAK-D Videos\2_1'
]

data = []
# out_folder = r'D:\OAK-D Videos'

for folder in tqdm(folder_list):
    vpaths= glob.glob(os.path.join(folder, '*.mp4'))
    for vpath in vpaths:
        if 'right' in vpath or 'left' in vpath:
        # if 'color' in vpath:
            get_frames(vpath) #, out_folder
            status = os.stat(vpath)
            modified = datetime.fromtimestamp(status.st_mtime, tz=timezone.utc)
            data.append([vpath,str(modified)])

# print(data)
# df = pd.DataFrame(data,columns=['vpath','datetime'])
# df.to_csv(r'C:\Users\dmchacon\Desktop\videos_info.csv',index=False)