import pandas as pd
import os
import numpy as np
from zipfile import ZipFile
from io import BytesIO


zip_file = ZipFile(os.path.join(os.path.expanduser('~'), 'Downloads/rainfall_NW_2016_02.3.npz.zip'))
for name in zip_file.namelist():
    b = BytesIO(zip_file.read(name))
    b.seek(0)
    npz = np.load(b, allow_pickle=True)
    df = pd.DataFrame.from_records([{item: npz[item] for item in npz.files}])

pass