import pandas as pd
import sys
import os

data_path = sys.argv[1]
world_size = int(sys.argv[2])

zinc = pd.read_csv(data_path)
n = zinc.shape[0]
m = n//world_size
path_new_folder='/raid/hsirelkhatim/data_per_rank'
if not os.path.exists(path_new_folder):
    os.mkdir(path_new_folder)
else:
    path, dirs, files = next(os.walk(path_new_folder))
    file_count = len(files)
    if file_count!=world_size:
        for file in files:
            os.remove(path_new_folder+'/'+file)
        for i in range(world_size):
            zinc.iloc[i:(i+1)*m,:].to_csv('/raid/hsirelkhatim/data_per_rank/{}.csv'.format(i))
    else:
        pass

