from matplotlib import pyplot as plt
import numpy as np
import h5py


if __name__=='__main__':
    list_of_points = []
    list_of_labels = []
    root = '/home/taosiyuan/cudaCode/CUDA-NN/data/splits'
    split = 'test'

    with h5py.File(f"{root}/{split}_point_clouds.h5","r") as hf:
        for k in hf.keys():
            list_of_points.append(hf[k]["points"][:].astype(np.float32))
            list_of_labels.append(hf[k].attrs["label"])

    idx = 5

    print(list_of_labels[idx], list_of_points[idx], sep='\n')

    

    # 创建三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [i[0] for i in list_of_points[idx]]
    y = [i[1] for i in list_of_points[idx]]
    z = [i[2] for i in list_of_points[idx]]
    ax.scatter(x, y, z)

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=0, azim=0)

    plt.savefig('point.jpg')
    