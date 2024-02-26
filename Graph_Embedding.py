import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
import joblib
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

def Divide_Kpca(x):

    list_data = []
    x = x.cpu().detach().numpy()
    x = x.reshape(-1, 1)    # (80000, 1)
    # 数据标准化
    scaler = StandardScaler()

    # 分批进行数据升维处理
    new_x = np.array_split(x, 100)
    for i in range(len(new_x)):
        data = new_x[i]
        data_standar = scaler.fit_transform(data)
        kpca = KernelPCA(kernel="rbf", gamma=0.1, n_components=2)
        data_kpca = kpca.fit_transform(data_standar)
        list_data.append(data_kpca)
    print(len(list_data))

    result = np.row_stack(list_data)
    print(result.shape)

    return result

def Kmeans(x):

    x = Divide_Kpca(x)

    # 分批进行K均值处理
    kmeans = MiniBatchKMeans(n_clusters=2, n_init='auto')
    kmeans.fit(x)
    joblib.dump(kmeans, 'kmeans_model.joblib')
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return labels, centers

if __name__ == "__main__":
    # y = torch.randn(32, 50, 50, 2).reshape(-1, 2)
    # model = joblib.load('E:\github files\CENew\CENet-main\CENet-main\kmeans_model.joblib')
    # adj_matrix = model.fit_predict(y.cpu().detach().numpy())
    # print(adj_matrix.shape)
    # exit()
    print("###########################")
    x = torch.load('adj_matrix')
    result = Kmeans(x)
    print("###########################")



