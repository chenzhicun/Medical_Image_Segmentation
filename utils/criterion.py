import torch
from torch.autograd import Function
import numpy as np
import copy
from queue import Queue


def IOU(predict, target):
    '''
    compute the intesection over union (IOU) of predict segmentation and ground truth segmentation.
    predict.shape: 1 X 1 X H X W
    target.shape: 1 X 1 X H X W
    '''
    intersection = torch.dot(predict.view(-1), target.view(-1))
    union = torch.sum(predict) + torch.sum(target) - intersection

    return intersection / union


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def accuracy(predict, gt):
    '''
    predict.shape: B X 1 X H X W
    gt.shape: B X 1 X H X W
    '''
    acc_tensor = predict == gt
    acc_tensor = acc_tensor.view(-1)
    return torch.sum(acc_tensor) / len(acc_tensor)


def vrand_vinfo(predict, label):
    def get_segment(data):
        data = copy.deepcopy(data)
        segment = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j]:
                    data[i][j] = False
                    temp_seg = [(i, j)]
                    task_queue = Queue()
                    task_queue.put((i, j))
                    while not task_queue.empty():
                        task = task_queue.get()
                        (m, n) = task
                        task_neighbor = [(m+1, n), (m+1, n-1), (m+1, n+1),
                                (m-1, n), (m-1, n-1), (m-1, n+1),
                                (m, n+1), (m, n-1)]
                        for neighbor in task_neighbor:
                            (x, y) = neighbor
                            if 0 <= x < 512 and 0 <= y < 512 and data[x][y]:
                                temp_seg.append((x, y))
                                data[x][y] = False
                                task_queue.put((x, y))
                    segment.append(temp_seg)
        return segment

    seg = get_segment(label)
    seg_predict = get_segment(predict)

    prob_matrix = np.zeros((len(seg_predict), len(seg)))
    for i in range(len(seg_predict)):
        for j in range(len(seg)):
            prob_matrix[i][j] = len(set(seg_predict[i]) & set(seg[j])) * 1.0 / 512 / 512

    s = np.sum(prob_matrix, axis=1)
    t = np.sum(prob_matrix, axis=0)

    prob_matrix_pow = prob_matrix * prob_matrix
    s_pow = s * s
    t_pow = t * t

    V_rand = np.sum(prob_matrix_pow.flatten()) / ( 0.5 * np.sum(s_pow) + 0.5 * np.sum(t_pow))

    prob_matrix_entropy = np.nan_to_num(prob_matrix * np.log(prob_matrix))
    s_entropy = np.nan_to_num(s * np.log(s))
    t_entropy = np.nan_to_num(t * np.log(t))

    I_s_t = np.sum(prob_matrix_entropy.flatten()) - np.sum(s_entropy) - np.sum(t_entropy)
    H_s = -np.sum(s_entropy)
    H_t = -np.sum(t_entropy)

    V_info = I_s_t / (0.5 * H_s + 0.5 * H_t)

    return V_rand, V_info
