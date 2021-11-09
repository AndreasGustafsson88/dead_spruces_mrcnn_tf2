"""Main module for snags custom config and dataset"""

import os
import json
import numpy as np
import skimage.draw
from typing import Tuple
from mrcnn_tf2.utilities import utils


class SnagsDataset(utils.Dataset):
    """
    Class for custom dataset
    """

    def read_image(self, annotations, dataset_dir: str, augmented) -> None:
        """Reads each image , extracts annotations and adds to self.add_image"""

        name_dict = {'snag': 1}

        if augmented:
            for file_name, polygons in annotations.items():
                objects = [poly['name'] for poly in polygons]

                num_ids = [name_dict[a] for a in objects]

                image_path = os.path.join(dataset_dir, file_name)
                image = skimage.io.imread(image_path)

                height, width = image.shape[:2]

                self.add_image(
                    'snag',
                    image_id=file_name,
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons,
                    num_ids=num_ids
                )
        else:
            annotations = [a for a in annotations.values()]
            for a in annotations:

                polygons = [r['shape_attributes'] for r in a['regions']]
                objects = [s['region_attributes']['name'] for s in a['regions']]

                num_ids = [name_dict[a] for a in objects]

                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)

                height, width = image.shape[:2]

                self.add_image(
                    'snag',
                    image_id=a['filename'],
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons,
                    num_ids=num_ids
                )

    def load_snags(self, dataset_dir: str, subset: str, augmented=False) -> None:
        """Method for loading custom dataset"""

        self.add_class('snag', 1, 'snag')
        assert subset in ['train', 'val']

        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, f"main_elov_{subset}_json.json")))

        self.read_image(annotations, dataset_dir, augmented)


    def load_mask(self, image_id: int) -> Tuple[np.array, np.array]:
        """
        Creates each binary mask for every mask in image based on annotations

        For this annotation only polygon values.
        """

        image_info = self.image_info[image_id]
        if image_info['source'] != 'snag':
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]

        if info['source'] != 'snag':
            return super(self.__class__, self).load_mask(image_id)

        num_ids = info['num_ids']
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])],
                        dtype=np.uint8)
        for i, p in enumerate(info['polygons']):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)

        return mask, num_ids

    def image_reference(self, image_id: int) -> str:

        info = self.image_info[image_id]

        if info['source'] == 'snag':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)
