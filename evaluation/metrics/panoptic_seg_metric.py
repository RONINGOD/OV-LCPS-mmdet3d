# Copyright (c) OpenMMLab. All rights reserved.
# Modified from mmdetection3d.
from typing import Dict, List, Optional

from mmengine.logging import MMLogger
import mmengine
from ..functional.panoptic_seg_eval import panoptic_seg_eval
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import SegMetric
import os
import json
import numpy as np


@METRICS.register_module()
class _PanopticSegMetric(SegMetric):
    def __init__(self,
                 class_names: List[str],
                 thing_class_inds: List[int],
                 stuff_class_inds: List[int],
                 thing_novel_class_inds: List[int],
                 stuff_novel_class_inds: List[int],
                 min_num_points: int,
                 id_offset: int,
                 dataset_type: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 taskset: str = None,
                 learning_map_inv = None, 
                 **kwargs):
        self.thing_class_inds = thing_class_inds # [0, 1, 2, 3, 4, 5, 6, 7]
        self.stuff_class_inds = stuff_class_inds # [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.min_num_points = min_num_points # 50
        self.id_offset = id_offset # 65536
        self.dataset_type=dataset_type # 'semantickitti'
        self.taskset = taskset 
        self.learning_map_inv = learning_map_inv # {0: 10, 1: 11, 2: 15, 3: 18, 4: 20, 5: 30, 6: 31, 7: 32, 8: 40, 9: 44, 10: 48, 11: 49, 12: 50, 13: 51, ...}
        self.novel_class_inds = thing_novel_class_inds+stuff_novel_class_inds
        self.thing_novel_class_inds = thing_novel_class_inds
        self.stuff_novel_class_inds = stuff_novel_class_inds
        self.classes = class_names
        for i in self.novel_class_inds:
            self.classes[i] = self.classes[i] + ' (novel)'

        super(_PanopticSegMetric, self).__init__(
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            prefix=prefix,
            collect_device=collect_device,
            **kwargs)

    # TODO modify format_result for panoptic segmentation evaluation, \
    # different datasets have different needs.

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix: # will report a error when finishing.
            self.format_results(results)
            return None
        # {0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4: 'bus', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist', 8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground', 12: 'building', 13: 'fence', ...}
        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index'] # 19
        # classes = self.dataset_meta['classes'] # ['car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', ...]
        classes = self.classes
        thing_classes = [classes[i] for i in self.thing_class_inds] # ['car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist', 'motorcyclist']
        stuff_classes = [classes[i] for i in self.stuff_class_inds] # ['road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign']
        include = self.thing_class_inds + self.stuff_class_inds # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]
        thing_novel_classes = [classes[i] for i in self.thing_novel_class_inds]
        stuff_novel_classes = [classes[i] for i in self.stuff_novel_class_inds]
        gt_labels = []
        seg_preds = []
        for eval_ann, sinlge_pred_results in results:
            gt_labels.append(eval_ann)
            seg_preds.append(sinlge_pred_results)

        ret_dict = panoptic_seg_eval(gt_labels, seg_preds, classes,
                                     thing_classes, stuff_classes, thing_novel_classes, stuff_novel_classes,
                                     include, self.dataset_type,
                                     self.min_num_points, self.id_offset,
                                     label2cat, ignore_index, logger)

        return ret_dict

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        mmengine.mkdir_or_exist(submission_prefix)
        ignore_index = self.dataset_meta['ignore_index']


        if self.dataset_type == 'nuscenes':
            meta_dir = os.path.join(submission_prefix, self.taskset)
            mmengine.mkdir_or_exist(meta_dir)
            meta =  {"meta": {
                "task": "segmentation",
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False}}
            output = open(os.path.join(submission_prefix, self.taskset, 'submission.json'), 'w')
            json_meta = json.dumps(meta)
            output.write(json_meta)
            output.close()

            for i, (eval_ann, result) in enumerate(results):
                sample_token = eval_ann['token']
                results_dir = os.path.join(submission_prefix, 'panoptic', self.taskset)
                mmengine.mkdir_or_exist(results_dir)
                results_dir = os.path.join(results_dir, sample_token) 
                pred_semantic_mask = result['pts_semantic_mask']
                pred_instance_mask = result['pts_instance_mask']
                pred_panoptic_mask = (pred_instance_mask + pred_semantic_mask*self.id_offset).astype(np.uint16)
                curr_file = results_dir +  "_panoptic.npz"
                np.savez_compressed(curr_file, data=pred_panoptic_mask)
        elif self.dataset_type == "semantickitti":
            results_dir = os.path.join(submission_prefix, 'sequences')
            mmengine.mkdir_or_exist(results_dir)
            for i in range(0,22):
                sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
                mmengine.mkdir_or_exist(sub_dir)

            learning_map_inv_array = np.zeros(len(self.learning_map_inv))
            for i, v in enumerate(self.learning_map_inv.values()):
                learning_map_inv_array[i] = v
            for i, (eval_ann, result) in enumerate(results):
                semantic_preds = result['pts_semantic_mask']
                instance_preds = result['pts_instance_mask']
                semantic_preds_inv = learning_map_inv_array[semantic_preds]
                panoptic_preds = semantic_preds_inv.reshape(-1, 1) + (instance_preds * self.id_offset).reshape(-1, 1)

                lidar_path = eval_ann['lidar_path']
                seq = lidar_path.split('/')[-3]
                pts_fname = lidar_path.split('/')[-1].split('.')[-2]+'.label'
                fname = os.path.join(results_dir, seq, 'predictions', pts_fname)
                panoptic_preds.reshape(-1).astype(np.uint32).tofile(fname)
        else:
            raise NotImplementedError()