import numpy as np

from fern import Fern
from utils import calculate_covariance
from utils import project_shape, similarity_transform


class FernCascade:
    def train(self, 
            images,
            current_shapes,
            ground_truth_shapes,
            bounding_box,
            mean_shape,
            second_level_num: int,
            candidate_pixel_num: int,
            fern_pixel_num: int,
            curr_level_num: int, 
            first_level_num: int):
        candidate_pixel_locations = np.zeros((candidate_pixel_num, 2))
        nearest_landmark_index = np.zeros((candidate_pixel_num, 1))
        self._second_level_num = second_level_num
        regression_targets = np.zeros((current_shapes.shape[0], ground_truth_shapes.shape[1], 2))

        for i in range(current_shapes.shape[0]):
            regression_targets[i] = project_shape(ground_truth_shapes[i, :], bounding_box[i]) - project_shape(current_shapes[i, :], bounding_box[i])
            scale, rotation = similarity_transform(mean_shape, project_shape(current_shapes[i, :],bounding_box[i]))
            regression_targets[i] = scale * np.dot(regression_targets[i], rotation.T)

        for i in range(candidate_pixel_num):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y > 1:
                i -= 1
                continue
            min_dist = 1e10
            min_index = 0
            for j in range(mean_shape.shape[0]):
                temp = (mean_shape[j, 0] - x) ** 2 + (mean_shape[j, 1] - y) ** 2
                if temp < min_dist:
                    min_dist = temp
                    min_index = j
            candidate_pixel_locations[i, 0] = x - mean_shape[min_index, 0]
            candidate_pixel_locations[i, 1] = y - mean_shape[min_index, 1]
            nearest_landmark_index[i] = min_index
        
        densities = [[] for _ in range(candidate_pixel_num)]

        for i in range(len(images)):
            temp = project_shape(current_shapes[i, :], bounding_box[i])
            scale, rotation = similarity_transform(temp, mean_shape)
            for j in range(candidate_pixel_num):
                index = int(nearest_landmark_index[j, 0])
                project_xy = scale * (np.sum(rotation * candidate_pixel_locations[j, :], axis=1)) * bounding_box[i].box_shape / 2.0
                project_xy += current_shapes[i, index, :]
                project_xy = project_xy.astype(np.int64)
                project_xy[0] = max(0, min(project_xy[0], images[i].shape[0] - 1))
                project_xy[1] = max(0, min(project_xy[1], images[i].shape[1] - 1))
                densities[j].append(images[i][tuple(project_xy)])

        covariance = np.zeros((candidate_pixel_num, candidate_pixel_num))
        for i in range(candidate_pixel_num):
            for j in range(candidate_pixel_num):
                covariance[i, j] = covariance[j, i] = calculate_covariance(densities[i], densities[j])

        prediction = np.zeros((len(regression_targets), mean_shape.shape[0], 2))
        self._ferns = [Fern() for _ in range(self._second_level_num)]
        for i in range(self._second_level_num):
            temp = self._ferns[i].train(densities, covariance, candidate_pixel_locations, nearest_landmark_index, regression_targets, fern_pixel_num)
            for j in range(len(temp)):
                curr = temp[j]
                prediction[j] += np.array(curr).reshape((29, 2))
                regression_targets[j] -= np.array(curr).reshape((29, 2))
            if (i + 1) % 50 == 0:
                print(f"Fern cascades: {curr_level_num} out of {first_level_num}; Ferns {i + 1} out of {second_level_num}")
        for i in range(prediction.shape[0]):
            scale, rotation = similarity_transform(project_shape(current_shapes[i, :], bounding_box[i]), mean_shape)
            prediction[i, :] = scale * np.dot(prediction[i, :], rotation.T)
        return prediction

    def predict(self, image, bounding_box, mean_shape, shape):
        result = np.zeros((shape.shape[0], 2))
        scale, rotation = similarity_transform(project_shape(shape, bounding_box), mean_shape)

        for i in range(len(self._ferns)):
            result += self._ferns[i].predict(image, shape, rotation, bounding_box, scale)
        scale, rotation = similarity_transform(project_shape(shape, bounding_box), mean_shape)

        return scale * np.dot(result, rotation.T)