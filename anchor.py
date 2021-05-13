import numpy as np
from sklearn.cluster import KMeans

def main():
    with open('train.txt', 'r') as f:
    lines = f.readlines()
    f.close()

    x = []

    for line in lines:
    data = line.split()
    img_path = data.pop(0)
    for d in data:
        x0, y0, x1, y1, _ = list(map(lambda x:int(x), d.split(',')))
        x.append([x1-x0, y1-y0])

    x = np.array(x)

    cls = KMeans(n_clusters=9)
    pred = cls.fit_predict(x)

    anchors = cls.cluster_centers_

    anchors = np.asarray(anchors, dtype='int')
    anchors = np.sort(anchors, axis=0)

    print(anchors)

if __name__ == '__name__':
    main()