import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.boxes as bops
from time import perf_counter as time
from pprint import pprint

@torch.no_grad()
def iou(bbox1, bbox2):
	bbox1 = bbox1.clone().detach()
	bbox2 = bbox2.clone().detach()
	bbox1 += 0.000000001
	bbox2 -= 0.000000001
	bb1w = bbox1[:, 2].clone()
	bb1h = bbox1[:, 3].clone()
	bb2w = bbox2[:, 2].clone()
	bb2h = bbox2[:, 3].clone()
	bbox1[:, 2] = bbox1[:, 0] + bb1w / 2
	bbox1[:, 0] = bbox1[:, 0] - bb1w / 2
	bbox1[:, 3] = bbox1[:, 1] + bb1h / 2
	bbox1[:, 1] = bbox1[:, 1] - bb1h / 2

	bbox2[:, 2] = bbox2[:, 0] + bb2w / 2
	bbox2[:, 0] = bbox2[:, 0] - bb2w / 2
	bbox2[:, 3] = bbox2[:, 1] + bb2h / 2
	bbox2[:, 1] = bbox2[:, 1] - bb2h / 2
	#convert to [x1, y1, x2, y2]
	out = torch.diagonal(bops.box_iou(bbox1, bbox2)).unsqueeze(0)
	#out = out.nan_to_num(1)
	return out

class YoloLoss(nn.Module):
	def __init__(self, lambda_coord=5, lambda_noobj=0.5, S=7, B=2, C=20):
		super().__init__()
		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj
		self.S = S
		self.B = B
		self.C = C
		self.mse = nn.MSELoss(reduction='sum')

	def forward(self, predictions, labels):
		''' 
		ONLY WORKS WITH 2 BOUNDING BOXES FOR NOW WILL CHANGE LATER
		# predictions of shape (B, S, S, (B * 5 + C))
		# Last dim is [x, y, w, h, Pr(c), x2, y2, w2, h2, Pr(c2), Pr(C1), Pr(C2) ... Pr(Cn)]
		'''

		object_inds = labels[..., 4] == 1

		obj_preds = predictions[object_inds]
		obj_labels = labels[object_inds]

		noobj_preds = predictions[~object_inds]
		noobj_labels = predictions[~object_inds]

		# Coordinate losses precalculation

		box1_pred = obj_preds[..., 0:5]
		box2_pred = obj_preds[..., 5:10]
		box_target = obj_labels[..., 0:5]

		ioub1 = iou(box1_pred[:, :4], box_target[:, :4]) # Index to not include class
		ioub2 = iou(box2_pred[:, :4], box_target[:, :4])

		best_boxes = torch.where((ioub1 > ioub2).reshape(-1, 1), box1_pred, box2_pred)

		# Loss for bounding box centers in cells with objects

		bbox_center_loss = self.mse(best_boxes[:, :2], box_target[:, :2])

		# Loss for bounding box dimensions in cells with objects

		bbox_dim_loss = self.mse(torch.sqrt(best_boxes[:, 2:4]), torch.sqrt(box_target[:, 2:4]))

		# Loss for confidence in cells with objects

		obj_conf_loss = self.mse(best_boxes[:, 4], box_target[:, 4])

		# Loss for confidence in cells without objects

		noobj_pred_conf = noobj_preds[:, [4, 9]] # will change to torch.arange(4, 5 * self.B, 5) later
		noobj_label_conf = noobj_labels[:, [4, 4]] # just [4] will work but pytorch gives broadcasting warning so whatever

		noobj_conf_loss = self.mse(noobj_pred_conf, noobj_label_conf)

		# Loss for classification in cells with objects

		classification_loss = self.mse(obj_preds[:, 10:], obj_labels[:, 10:])

		# FINAL LOSS

		coord_loss = self.lambda_coord * (bbox_center_loss + bbox_dim_loss)
		confidence_loss = obj_conf_loss + self.lambda_noobj * noobj_conf_loss

		loss = (coord_loss + confidence_loss + classification_loss) / predictions.shape[0]

		return loss



