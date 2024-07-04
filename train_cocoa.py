#!/usr/bin/env python
"""
A main training script, based on detectron2 tools/train_net.py

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader

# aistron
from aistron.data import AmodalDatasetMapper
from aistron.evaluation import AmodalInstanceEvaluator
from aistron.config import add_aistron_config

from aistron.data.datasets.coco_amodal import register_aistron_cocolike_instances


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["coco_amodal"]:
        evaluator_list.append(AmodalInstanceEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AmodalDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = AmodalDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_aistron_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    register_aistron_cocolike_instances(
        name ="annotation_cocoa_train",
        metadata ={},
        json_file = "/content/data/datasets/COCOA/annotations/annotations_aistron.json",
        image_root ="/content/data/datasets/COCOA/annotations")
    register_aistron_cocolike_instances(
        name ="annotation_cocoa_test",
        metadata ={},
        json_file = "/content/data/datasets/COCOA/annotations/annotations_aistron.json",
        image_root ="/content/data/datasets/COCOA/annotations")
    # register_aistron_cocolike_instances(
    #     name ="annotation_roboflow_train",
    #     metadata ={},
    #     json_file = "/media/binh/D/ComputerVision/data/datasets/COCOA/annotations/annotations_aistron.json",
    #     image_root ="/media/binh/D/ComputerVision/data/datasets/COCOA/roboflow/train")

    # register_aistron_cocolike_instances(
    #     name ="annotation_roboflow_valid",
    #     metadata ={},
    #     json_file = "/media/binh/D/ComputerVision/data/datasets/COCOA/roboflow/valid/_annotations.coco.json",
    #     image_root ="/media/binh/D/ComputerVision/data/datasets/COCOA/roboflow/valid")

    # register_aistron_cocolike_instances(
    #     name ="annotation_roboflow_test",
    #     metadata ={},
    #     json_file = "/media/binh/D/ComputerVision/data/datasets/COCOA/roboflow/test/_annotations.coco.json",
    #     image_root ="/media/binh/D/ComputerVision/data/datasets/COCOA/roboflow/test")

    # register_aistron_cocolike_instances(
    #     name = "annotation_cocoa_amodal",
    #     metadata = {},
    #     json_file = "/media/binh/D/ComputerVision/data/datasets/COCOA/annotations/annotations_aistron.json",
    #     image_root = "/media/binh/D/ComputerVision/data/datasets/COCOA/annotations")


    cfg.OUTPUT_DIR = "/content/data/train_outputs"
    cfg.DATASETS.TRAIN = ("annotation_cocoa_train",)
    #cfg.DATASETS.VAL = ("annotation_roboflow_valid",)
    cfg.DATASETS.TEST = ("annotation_cocoa_test",)
    cfg.TEST.AUG.ENABLED = True
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# Hàm load ảnh từ dataset, nhận diện tinh thể trong ảnh và hiển thị
# cùng với ảnh được dán nhãn thủ công bằng tay
import random
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from IPython.display import display, Image
from detectron2.data import MetadataCatalog, DatasetCatalog
import logging
from detectron2.engine import DefaultPredictor
from PIL import Image
import time
import torch
test_dataset = "annotation_cocoa_test"
def show_demo_inference_image_from_dataset(dataset_dicts, cfg,sample_size = 1):
  predictor = DefaultPredictor(cfg)
  test_metadata = MetadataCatalog.get(test_dataset)
  i=0
  for d in random.sample(dataset_dicts, sample_size):
    img_origin = cv2.imread(d["file_name"])
    #img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    outputs = predictor(img_origin)
    end_time = time.time()
    v = Visualizer(img_origin[:, :, ::-1], metadata=test_metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print(f"Inferred Time: {end_time - start_time}")
    visualizer = Visualizer(img_origin[:, :, ::-1], metadata=test_metadata, scale=0.8)
    vis = visualizer.draw_dataset_dict(d)

    Hori = np.concatenate((out.get_image()[:, :, ::-1], vis.get_image()[:, :, ::-1]), axis=1)
    Hori_pil = Image.fromarray(Hori)
    #Save image
    Hori_pil.save("test_" + str(i) + ".jpg")

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        dataset_dictionaries = DatasetCatalog.get(test_dataset)
        show_demo_inference_image_from_dataset(dataset_dictionaries,cfg)

        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    trainer.train()
    dataset_dictionaries = DatasetCatalog.get(test_dataset)
    show_demo_inference_image_from_dataset(dataset_dictionaries,cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )