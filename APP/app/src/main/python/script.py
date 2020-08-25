import torch
import numpy as np

from math import ceil
from itertools import product as product

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def to_cuda(elements, device):
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.to(device) for x in elements]
        return elements.to(device)
    return elements


def batched_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [N, num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    priors = priors[None]
    boxes = torch.cat((
        priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :,  2:],
        priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])),
        dim=2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def JtoTensor(Jarray, columns):
    temp_list = list()

    for item in Jarray:
        temp_list.append(item)

    dimension = int(len(temp_list)/columns)

    temp_tensor = torch.Tensor(temp_list).view(-1, columns).unsqueeze(dim=1).view(1, dimension, columns)
    return temp_tensor

# TODO: carried input is not suitable for below function
def locate_faces(bboxes):
    temp_list = list()

    for bbox in bboxes:
        x_min, y_min, x_max, y_max, conf = [int(_) for _ in bbox]
        print("draw_faces : {}, {}, {}, {}".format(x_min, y_min, x_max, y_max))

        temp_list.append(x_min)
        temp_list.append(y_min)
        temp_list.append(x_max)
        temp_list.append(y_max)

    return temp_list


def detect(loc, conf, height, width):
    scores = conf[:, :, 1:]
    priorbox = PriorBox(cfg_mnet, image_size=(height, width))
    priors = priorbox.forward()
    priors = to_cuda(priors, torch.device("cpu"))
    prior_data = priors.data
    boxes = batched_decode(loc, prior_data, cfg_mnet['variance'])
    boxes = torch.cat((boxes, scores), dim=-1)

    # TODO: matrix manipulation should be done here!!
    # boxes ==> torch.Size([1, 28928, 5])
    # desired Result.shape ==> (4, 5) np.ndarray

    return boxes


def main(locJArray, confJArray, height, width):
    loc = JtoTensor(locJArray, 4) # torch.Size([1, 28928, 4])
    conf = JtoTensor(confJArray, 2) # torch.Size([1, 28928, 2])

    detections = detect(loc, conf, height, width).numpy()
    detections = np.squeeze(detections[:, :4], axis=0)
    # desired detections.shape ==> (4, 4) np.ndarray

    # number of instance
    num_of_detected_faces = detections.shape[0]

    return num_of_detected_faces

