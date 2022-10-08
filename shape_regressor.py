import numpy as  np

from fern_cascade import FernCascade
from utils import get_mean_shape, project_shape, reproject_shape


class ShapeRegressor():
    def __init__(self) -> None:
        self._first_level_num = 0

    def train(self,
            images, 
            ground_truth_shapes,
            bounding_box,
            first_level_num: int,
            second_level_num: int,
            candidate_pixel_num: int,
            fern_pixel_num: int,
            initial_num: int):
        print("Start training...")
        self._bounding_box = bounding_box
        self._training_shapes = ground_truth_shapes
        self._first_level_num
        self._landmark_num = ground_truth_shapes.shape[1]
        augmented_images = []
        augmented_bounding_box = []
        augmented_ground_truth_shapes = np.zeros((len(images) * initial_num, ground_truth_shapes.shape[1], ground_truth_shapes.shape[2]))
        current_shapes = np.zeros((len(images) * initial_num, self._landmark_num, 2))

        k = 0
        for i in range(len(images)):
            for j in range(initial_num):
                if k > len(images) * initial_num:
                    print(k)
                index = int(np.random.uniform(0, len(images) - 1))
                while index == i:
                    index = int(np.random.uniform(0, len(images) - 1))
                augmented_images.append(images[i])
                augmented_ground_truth_shapes[k, :] = ground_truth_shapes[i, :]
                augmented_bounding_box.append(bounding_box[i])

                temp = ground_truth_shapes[index, :]
                temp = project_shape(temp, bounding_box[index])
                temp = reproject_shape(temp, bounding_box[i])
                current_shapes[k, :] = temp
                k += 1
        
        self._mean_shape = get_mean_shape(ground_truth_shapes, bounding_box)
        self._fern_cascades = [FernCascade() for _ in range(first_level_num)]
        for i in range(first_level_num):
            print(f"Training {i + 1} cascade out of {first_level_num}")
            prediction = self._fern_cascades[i].train(augmented_images, current_shapes, augmented_ground_truth_shapes, augmented_bounding_box, self._mean_shape, second_level_num, candidate_pixel_num, fern_pixel_num, i+1, first_level_num)
            for j in range(prediction.shape[0]):
                temp = prediction[j] + project_shape(current_shapes[j, :], augmented_bounding_box[j])
                current_shapes[j, :] = reproject_shape(current_shapes[j, :], augmented_bounding_box[j])
        

    def predict(self, image, bounding_box, initial_num):
        result = np.zeros((self._landmark_num, 2))
        for _ in range(initial_num):
            index = int(np.random.uniform(0, self._training_shapes.shape[0]))
            current_shape = self._training_shapes[index]
            current_bounding_box = self._bounding_box[index]
            current_shape = project_shape(current_shape, current_bounding_box)
            current_shape = reproject_shape(current_shape, current_bounding_box)
            for j in range(self._first_level_num):
                prediction = self._fern_cascades[j].predict(image, bounding_box, self._mean_shape, current_shape)
                current_shape = prediction + project_shape(current_shape, current_bounding_box)
                current_shape = prediction + reproject_shape(current_shape, current_bounding_box)

            result += current_shape

        return result / initial_num