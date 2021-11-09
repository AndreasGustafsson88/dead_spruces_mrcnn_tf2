from mrcnn_tf2.models.snags import SnagsDataset
from mrcnn_tf2.models.custom_config import TrainingSnagsConfig, DEFAULT_LOGS_DIR, COCO_WEIGHTS_PATH, DATASET_DIR
from mrcnn_tf2.models import model as modellib
import warnings
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser(description='Train Mask R-CNN with tensorflow 2.')

    parser.add_argument('--dataset_dir', '-d', help='Path to dataset', type=str, default=DATASET_DIR)

    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    config = TrainingSnagsConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="training",
        config=config,
        model_dir=DEFAULT_LOGS_DIR
        )

    model.load_weights(
        COCO_WEIGHTS_PATH,
        by_name=True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    )

    # Load training dataset
    dataset_train = SnagsDataset()
    dataset_train.load_snags(DATASET_DIR if args.dataset_dir != DATASET_DIR else args.dataset_dir, 'train', augmented=False)
    dataset_train.prepare()

    # Load validation dataset
    dataset_val = SnagsDataset()
    dataset_val.load_snags(DATASET_DIR if args.dataset_dir != DATASET_DIR else args.dataset_dir, 'val')
    dataset_val.prepare()

    model.train(
        dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=10,
        layers='heads'
    )