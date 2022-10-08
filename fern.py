from math import sqrt
import numpy as np

from bounding_box import BoundingBox
from utils import calculate_covariance


class Fern:
    def __init__(self) -> None:
        pass

    def train(self, candidate_pixel_intensity, covariance, candidate_pixel_locations, nearest_landmark_index, regression_targets, fern_pixel_num):
        self._fern_pixel_num = fern_pixel_num
        self._landmark_num = regression_targets.shape[1]
        self._selected_pixel_index = np.zeros((fern_pixel_num, 2), dtype=np.int64)
        self._selected_pixel_locations = np.zeros((fern_pixel_num, 4))
        self._selected_nearest_landmark_index = np.zeros((fern_pixel_num, 2), dtype=np.int64)
        candidate_pixel_num = candidate_pixel_locations.shape[0]
        self._threshold = np.zeros((fern_pixel_num, 1))

        for i in range(self._fern_pixel_num):
            random_direction = np.random.uniform(-1.1, 1.1, (self._landmark_num, 2))
            random_direction = random_direction / np.linalg.norm(random_direction)
            projection_result = np.zeros(regression_targets.shape[0])
            for j in range(regression_targets.shape[0]):
                projection_result[j] = np.sum(regression_targets[j] * random_direction)
            covariance_projection_density = np.zeros((candidate_pixel_num, 1))
            for j in range(candidate_pixel_num):
                covariance_projection_density[j] = calculate_covariance(projection_result, candidate_pixel_intensity[j])

            max_correlation = -1
            max_pixel_index_1 = 0
            max_pixel_index_2 = 0   
            for j in range(candidate_pixel_num):
                for k in range(candidate_pixel_num):
                    temp1 = covariance[j,j] + covariance[k,k] - 2*covariance[j,k];
                    if abs(temp1) < 1e-10:
                        continue
                    flag = False
                    for p in range(i):
                        if j == self._selected_pixel_index[p,0] and k == self._selected_pixel_index[p,1]:
                            flag = True
                            break 
                        elif j == self._selected_pixel_index[p,1] and k == self._selected_pixel_index[p,0]:
                            flag = True
                            break
                    if flag:
                        continue
                    temp = (covariance_projection_density[j] - covariance_projection_density[k]) / sqrt(temp1)
                    if abs(temp) > max_correlation:
                        max_pixel_index_1 = j
                        max_correlation = temp
                        max_pixel_index_2 = k
                
            self._selected_pixel_index[i,0] = max_pixel_index_1
            self._selected_pixel_index[i,1] = max_pixel_index_2
            self._selected_pixel_locations[i,0] = candidate_pixel_locations[max_pixel_index_1,0]
            self._selected_pixel_locations[i,1] = candidate_pixel_locations[max_pixel_index_1,1]
            self._selected_pixel_locations[i,2] = candidate_pixel_locations[max_pixel_index_2,0]
            self._selected_pixel_locations[i,3] = candidate_pixel_locations[max_pixel_index_2,1]
            self._selected_nearest_landmark_index[i,0] = nearest_landmark_index[max_pixel_index_1]
            self._selected_nearest_landmark_index[i,1] = nearest_landmark_index[max_pixel_index_2]

            max_diff =np.max(np.abs(np.array(candidate_pixel_intensity[max_pixel_index_1]) - np.array(candidate_pixel_intensity[max_pixel_index_2])))
            self._threshold[i] = np.random.uniform(-0.2*max_diff,0.2*max_diff)

        bin_num = 2 ** fern_pixel_num
        self._selected_pixel_index = self._selected_pixel_index.astype(np.int64)
        shapes_in_bin = [[] for _ in range(bin_num)]
        for i in range(regression_targets.shape[0]):
            index = 0
            for j in range(fern_pixel_num):
                density_1 = candidate_pixel_intensity[self._selected_pixel_index[j,0]][i]
                density_2 = candidate_pixel_intensity[self._selected_pixel_index[j,1]][i]
                if density_1 - density_2 >= self._threshold[j]:
                    index += 2 ** j
            shapes_in_bin[index].append(i)
        prediction = [[] for _ in range(regression_targets.shape[0])]
        self._bin_output = np.zeros((regression_targets.shape[0], self._landmark_num, 2))

        for i in range(bin_num):
            temp = np.zeros((self._landmark_num, 2))
            bin_size = len(shapes_in_bin[i])
            if bin_size == 0:
                self._bin_output[i] = temp
                continue
            for j in range(bin_size):
                temp += regression_targets[shapes_in_bin[i][j]]
            temp = (1.0/((1.0+1000.0/bin_size) * bin_size)) * temp
            self._bin_output[i] = temp
            for j in range(bin_size):
                prediction[shapes_in_bin[i][j]].append(temp) 
        return prediction


    def predict(self, image, shape, rotation, bounding_box: BoundingBox, scale: float):
        index = 0
        for i in range(self._fern_pixel_num):
            nearest_landmark_index_1 = self._selected_nearest_landmark_index[i,0]
            nearest_landmark_index_2 = self._selected_nearest_landmark_index[i,1]

            project_xy = scale * (np.sum(rotation * self._selected_pixel_locations[i, :2], axis=1)) * bounding_box.box_shape / 2.0 + shape[nearest_landmark_index_1, :]

            project_xy[0] = max(0.0, min(project_xy[0], image.shape[1] - 1))
            project_xy[1] = max(0.0, min(project_xy[1], image.shape[0] - 1))

            intensity_1 = image[tuple(project_xy)]

            project_xy = scale * (np.sum(rotation * self._selected_pixel_locations[i, 2:], axis=1)) * bounding_box.box_shape / 2.0 + shape[nearest_landmark_index_2, :]

            project_xy[0] = max(0.0, min(project_xy[0], image.shape[1] - 1))
            project_xy[1] = max(0.0, min(project_xy[1], image.shape[0] - 1))

            intensity_2 = image[tuple(project_xy)]

            if intensity_1 - intensity_2 >= self._threshold[i]:
                index = index + 2 ** i
        
        return self._bin_output[index]