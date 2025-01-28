import h5py
import numpy as np

def read_matlab_v73_files(file_path):
    file_dict = {}
    f = h5py.File(file_path)
    for k1, v1 in f.items():
        file_dict[k1] = np.array(v1)
        if type(f[k1]) == h5py._hl.group.Group:
            file_dict[k1] = {}
            for k2, v2 in f[k1].items():
                file_dict[k1][k2] = np.array(v2)
                if type(f[k1][k2]) == h5py._hl.group.Group:
                    file_dict[k1][k2] = {}
                    for k3, v3 in f[k1][k2].items():
                        file_dict[k1][k2][k3] = np.array(v3)
    return file_dict