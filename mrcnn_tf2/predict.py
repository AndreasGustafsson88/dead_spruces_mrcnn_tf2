from mrcnn_tf2.models.snags import SnagsDataset
from mrcnn_tf2.models.custom_config import InferenceSnagsConfig, TRAINED_MODELS, DATASET_DIR, DEFAULT_LOGS_DIR
from mrcnn_tf2.models import model as modellib
from mrcnn_tf2.utilities import utils, visualize
import numpy as np
import os


def get_results(image, image_meta, gt_class_id, gt_bbox, gt_mask, image_id):
    """
    Loop through all images and get ground truth values from validation dataset.
    """
    # info = [dataset.image_info[image] for image in image_id]

    for i in image_id:
        i, i_meta, gt, gt_b, gt_m = modellib.load_image_gt(dataset, config, image_id[i])
        image.append(i)
        image_meta.append(image_meta)
        gt_class_id.append(gt)
        gt_bbox.append(gt_b)
        gt_mask.append(gt_m)

    results = [model.detect([i]) for i in image]

    return [result[0] for result in results]


def run_inference(show_instances: bool = True, show_differences: bool = True, show_precision_recall: bool = True):

    """
    Run inference from validation dataset on trained model.
    """

    # Init lists and get all image_ids
    image, image_meta, gt_class_id, gt_bbox, gt_mask = [], [], [], [], []
    image_id = [id for id in dataset.image_ids]

    res = get_results(image, image_meta, gt_class_id, gt_bbox, gt_mask, image_id)

    if show_differences:
        for i in image_id:
            visualize.display_differences(image[i],
                                          gt_bbox[i],
                                          gt_class_id[i],
                                          gt_mask[i],
                                          res[i]["rois"],
                                          res[i]["class_ids"],
                                          res[i]["scores"],
                                          res[i]['masks'],
                                          dataset.class_names, title=f"Predictions{i}")

    if show_instances:
        for i in image_id:
            visualize.display_instances(image[i],
                                        res[i]['rois'],
                                        res[i]['masks'],
                                        res[i]['class_ids'],
                                        dataset.class_names,
                                        res[i]['scores'],
                                        title=f"Predictions{i}",
                                        colors=[(0, 1, 0, .8)] * len(res[i]['rois']))

    APs, precisions_dict, recall_dict, mAP_dict, f_1 = [], {}, {}, {}, {}

    for i in image_id:
        Ap, pre, rec, _ = utils.compute_ap(gt_bbox[i],
                                           gt_class_id[i],
                                           gt_mask[i],
                                           res[i]["rois"],
                                           res[i]["class_ids"],
                                           res[i]["scores"],
                                           res[i]['masks'])

        mAP_dict[dataset.image_info[image_id[i]]['id']] = np.mean(Ap)
        precisions_dict[dataset.image_info[image_id[i]]['id']] = np.mean(pre)
        recall_dict[dataset.image_info[image_id[i]]['id']] = np.mean(rec)
        APs.append(Ap)

        if show_precision_recall:
            visualize.plot_precision_recall(Ap, pre, rec)

    return np.mean(APs), precisions_dict, recall_dict, mAP_dict


if __name__ == '__main__':
    # Specify which model to run inference on
    model_name = 'mask_rcnn_snag_0020.h5'
    dir_name = '0806_20E'

    # Init Config for inference
    config = InferenceSnagsConfig()
    config.display()

    # Set path to model
    WEIGHTS_PATH = os.path.join(TRAINED_MODELS, dir_name, model_name)

    # Init dataset from val folder
    dataset = SnagsDataset()
    dataset.load_snags(DATASET_DIR, "val")
    dataset.prepare()

    # Init model and load weights from trained model
    model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR, config=config)
    model.load_weights(WEIGHTS_PATH, by_name=True)

    val_AP, precision, recall, mAP = run_inference(show_instances=True,
                                                   show_differences=True,
                                                   show_precision_recall=True)

    print(np.mean(val_AP))
    print(precision)
    print(recall)
    print(mAP)

