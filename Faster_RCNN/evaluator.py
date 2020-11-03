from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base import Base as DatasetBase
from model import Model
from collections import deque

class Evaluator(object):
    def __init__(self, dataset: DatasetBase, path_to_data_dir: str, path_to_results_dir: str):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self._path_to_data_dir = path_to_data_dir
        self._path_to_results_dir = path_to_results_dir

    def evaluate(self, model: Model) -> Tuple[float, str,float]:
        all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs = [], [], [], []
        losses = deque(maxlen=100)

        with torch.no_grad():
            #for _, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch) in enumerate(tqdm(self._dataloader)):
            for _, (image_id_batch, image_batch, scale_batch, bboxes_batch, labels_batch) in enumerate(tqdm(self._dataloader)):

                #print('labels_batch_evaluator',labels_batch)
                #print('bboxes_batch_evaluator',bboxes_batch)
                image_batch = image_batch.cuda()
                assert image_batch.shape[0] == 1, 'do not use batch size more than 1 on evaluation'

                detection_bboxes, detection_classes, detection_probs, detection_batch_indices = \
                    model.eval().forward(image_batch)
                #print('detection_bboxes_objectness',detection_bboxes)
                #print('detection_classes_transformer',detection_classes)
                scale_batch = scale_batch[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(device=detection_bboxes.device)
                detection_bboxes = detection_bboxes / scale_batch

                #kept_indices = (detection_probs > 0.05).nonzero().view(-1)
                kept_indices = (detection_probs > 0.00).nonzero().view(-1)
                detection_bboxes = detection_bboxes[kept_indices]
                detection_classes = detection_classes[kept_indices]
                detection_probs = detection_probs[kept_indices]
                detection_batch_indices = detection_batch_indices[kept_indices]

                all_detection_bboxes.extend(detection_bboxes.tolist())
                all_detection_classes.extend(detection_classes.tolist())
                all_detection_probs.extend(detection_probs.tolist())
                all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])
                

		
                #image_batch = image_batch.cuda()
                bboxes_batch = bboxes_batch.cuda()
                labels_batch = labels_batch.cuda()
                #print('before forward')
                
                #print('image_batch',image_batch)
                #print('bboxes_batch',bboxes_batch)
                #print('labels_batch',labels_batch)


                anchor_objectness_losses, anchor_transformer_losses, proposal_class_losses, proposal_transformer_losses = \
                    model.train().forward(image_batch, bboxes_batch, labels_batch)
                #print('anchor_objectness_losses_evaluator:',anchor_objectness_losses)
                #print('anchor_transformer_losses_evaluator:',anchor_transformer_losses)
                #print('proposal_class_losses_evaluator:',proposal_class_losses)
                #print('proposal_transformer_losses_evaluator:',proposal_transformer_losses)
                anchor_objectness_loss = anchor_objectness_losses.mean()
                anchor_transformer_loss = anchor_transformer_losses.mean()
                proposal_class_loss = proposal_class_losses.mean()
                proposal_transformer_loss = proposal_transformer_losses.mean()
                loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss
                losses.append(loss.item())
                

                all_image_ids.extend([image_id_batch[i] for i in detection_batch_indices])
        avg_loss_eval = sum(losses) / len(losses)
        mean_ap, detail= self._dataset.evaluate(self._path_to_results_dir, all_image_ids, all_detection_bboxes, all_detection_classes, all_detection_probs)
        return mean_ap, detail,avg_loss_eval
