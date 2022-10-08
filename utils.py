import numpy as np


def project_shape(shape, bounding_box):
    return (shape - bounding_box.centroid) / (bounding_box.box_shape / 2)

def reproject_shape(shape, bounding_box):
    return shape * bounding_box.box_shape / 2 + bounding_box.centroid

def get_mean_shape(shape, bounding_box):
    res = np.zeros((shape.shape[1], 2))
    for i in range(len(shape)):
        res += project_shape(shape[i], bounding_box[i])
    return res / shape.shape[0]

def similarity_transform(shape1, shape2):
    #print(shape1.shape, shape2.shape)
    temp1 = shape1 - np.mean(shape1, axis=0)
    temp2 = shape2 - np.mean(shape2, axis=0)
    #print(temp1.shape, temp2.shape)
    s1 = np.linalg.norm(np.cov(temp1.T))
    s2 = np.linalg.norm(np.cov(temp2.T))
    scale = s1 / s2
    temp1 = temp1 / s1
    temp2 = temp2 / s2
    num = np.sum(temp1[:, 1] * temp2[:, 0] - temp1[:, 0] * temp2[:, 1])
    den = np.sum(temp1[:, 0] * temp2[:, 0] - temp1[:, 1] * temp2[:, 1])
    #print(num.shape, den.shape)
    norm = np.sqrt(num * num + den * den)
    rotation = np.zeros((2, 2))
    rotation[0, 0] = rotation[1, 1] = den / norm
    rotation[1, 0] = num / norm
    rotation[0, 1] = - num / norm
    return scale, rotation

def calculate_covariance(v1, v2) -> float:
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.mean((v1 - np.mean(v1)) * (v2 - np.mean(v2)))

if __name__ == '__main__':
    a = np.array([[1, 2], [3, 4]])
    x = np.array([0, ])
    print(a[0, :2])
    print(np.zeros((1, 2)))
    print(np.random.uniform(0, 10))
    print((lambda x, y: x + y)(*{1: 'a', 2: 'b'}.items()))
    