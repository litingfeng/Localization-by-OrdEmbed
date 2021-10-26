# -*- coding: utf-8 -*-
# @Time : 9/9/21 7:29 AM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
# ref: https://github.com/otoofim/ObjLocalisation/blob/master/lib/Agent.py
# Defined actions in paper "tree-structured reinforcement learning for sequential object localization"

import numpy as np
import torch
import torch.nn.functional as F

ACTIONS = {
# Scaling
0: 'SCALE_TL', # shrink toward top left
1: 'SCALE_TR', # shrink toward top right
2: 'SCALE_BL', # shrink toward bottom left
3: 'SCALE_BR', # shrink toward bottom right
4: 'SCALE_DOWN', # shrink toward center
5: 'PLACE_LANDMARK',
# Translation
6: 'MOVE_LEFT',
7: 'MOVE_RIGHT',
8: 'MOVE_UP',
9: 'MOVE_DOWN',
10: 'ASPECT_RATIO_UP',  # enlarge H, keep W
11: 'ASPECT_RATIO_DOWN', # enlarge W, heep H
12: 'ASPECT_RATIO_UP_S', # reduce W, keep H
13: 'ASPECT_RATIO_DOWN_S' # reduce H, keep W
}
# Transformation coefficients
STEP_FACTOR = 0.2
MAX_ASPECT_RATIO = 3.00
MIN_ASPECT_RATIO = 0.3

class Actor:
    def __init__(self, img_size, batch_size, min_box_side=50):
        self.img_h, self.img_w = img_size
        self.min_box_side = min_box_side
        self.agent_windows = np.repeat(np.array([[0, 0, self.img_w-1, self.img_h-1]]),
                                       batch_size, axis=0)

    def take_action(self, actions):
        new_boxes = []
        for i, action in enumerate(actions):
            self.current_window = self.agent_windows[i]
            self.current_window = getattr(self, ACTIONS[action.item()].lower())()
            self.adjust()
            new_boxes.append(torch.from_numpy(self.current_window))
            self.agent_windows[i] = self.current_window

        return torch.stack(new_boxes).float()

    def scale_tl(self):
        """shrink toward top left
        """
        newbox = np.copy(self.current_window)
        change_w, change_h = self._get_scale_change(newbox)
        newbox[2], newbox[3] = newbox[2] - change_w, newbox[3] - change_h
        return newbox

    def scale_tr(self):
        """shrink toward top right
        """
        newbox = np.copy(self.current_window)
        change_w, change_h = self._get_scale_change(newbox)
        newbox[0], newbox[3] = newbox[0] + change_w, newbox[3] - change_h
        return newbox

    def scale_bl(self):
        """shrink toward bottom left
        """
        newbox = np.copy(self.current_window)
        change_w, change_h = self._get_scale_change(newbox)
        newbox[1], newbox[2] = newbox[1] + change_h, newbox[2] - change_w
        return newbox

    def scale_br(self):
        """shrink toward bottom left
        """
        newbox = np.copy(self.current_window)
        change_w, change_h = self._get_scale_change(newbox)
        newbox[0], newbox[1] = newbox[0] + change_w, newbox[1] + change_h
        return newbox

    def scale_down(self):
        """shrink toward center
        """
        newbox = np.copy(self.current_window)
        change_w, change_h = self._get_scale_change(newbox)
        change_w, change_h = change_w / 2., change_h / 2.
        newbox[0], newbox[1], newbox[2], newbox[3] = newbox[0] + change_w, newbox[1] + change_h, \
                                                     newbox[2] - change_w, newbox[3] - change_h
        return newbox

    def place_landmark(self):
        '''do nothing'''
        return self.current_window

    def move_left(self):
        newbox = np.copy(self.current_window)
        box_w = newbox[2] - newbox[0] + 1
        step = STEP_FACTOR * box_w
        if newbox[0] - step >= 0:
            newbox[0], newbox[2] = newbox[0] - step, newbox[2] - step
        else:
            newbox[0], newbox[2] = 0, box_w - 1

        return newbox

    def move_right(self):
        newbox = np.copy(self.current_window)
        box_w = newbox[2] - newbox[0] + 1
        step = STEP_FACTOR * box_w
        if newbox[2] + step < self.img_w:
            newbox[0], newbox[2] = newbox[0] + step, newbox[2] + step
        else:
            newbox[0], newbox[2] = self.img_w - box_w, self.img_w - 1

        return newbox

    def move_up(self):
        newbox = np.copy(self.current_window)
        box_h = newbox[3] - newbox[1] + 1
        step = STEP_FACTOR * box_h
        if newbox[1] - step >= 0:
            newbox[1], newbox[3] = newbox[1] - step, newbox[3] - step
        else:
            newbox[1], newbox[3] = 0, box_h - 1

        return newbox

    def move_down(self):
        newbox = np.copy(self.current_window)
        box_h = newbox[3] - newbox[1] + 1
        step = STEP_FACTOR * box_h
        if newbox[3] + step < self.img_h:
            newbox[1], newbox[3] = newbox[1] + step, newbox[3] + step
        else:
            newbox[1], newbox[3] = self.img_h - box_h, self.img_h - 1

        return newbox

    def aspect_ratio_up(self):
        """ increase aspect ratio, enclarge box_h, keep box_w
        """
        newbox = np.copy(self.current_window)
        box_w, box_h = self._get_box_wh(newbox)
        change_w, change_h = self._get_change(STEP_FACTOR, box_w, box_h)
        if box_h + change_h <= self.img_h:
            ar = (box_h + change_h) / float(box_w)
            newDelta = STEP_FACTOR if ar < MAX_ASPECT_RATIO else 0.
        else:
            newDelta = self.img_h / float(box_h) - 1
            ar = (box_h + newDelta * box_h) / float(box_w)
            if ar > MAX_ASPECT_RATIO:
                newDelta = 0.
        change_h = newDelta * box_h / 2.
        newbox[1], newbox[3] = newbox[1] - change_h - 0.5, newbox[3] + change_h + 0.5

        return newbox

    def aspect_ratio_down(self):
        """ decrease aspect ratio, enclarge box_w, keep box_h
        """
        newbox = np.copy(self.current_window)
        box_w, box_h = self._get_box_wh(newbox)
        change_w, change_h = self._get_change(STEP_FACTOR, box_w, box_h)
        if box_w + change_w <= self.img_w:
            ar = float(box_h) / (box_w + change_w)
            newDelta = STEP_FACTOR if ar < MAX_ASPECT_RATIO else 0.
        else:
            newDelta = self.img_w / float(box_w) - 1
            ar = float(box_h) / (box_w + newDelta * box_w)
            if ar > MAX_ASPECT_RATIO:
                newDelta = 0.
        change_w = newDelta * box_w / 2.
        newbox[0], newbox[2] = newbox[0] - change_w - 0.5, newbox[2] + change_w + 0.5

        return newbox

    def aspect_ratio_up_s(self):
        """ increase aspect ratio, reduce box_w, keep box_h
        """
        newbox = np.copy(self.current_window)
        box_w, box_h = self._get_box_wh(newbox)
        change_w, change_h = self._get_change(STEP_FACTOR, box_w, box_h)
        if box_w - change_w >= self.min_box_side:
            ar = float(box_h) / (box_w - change_w)
            newDelta = STEP_FACTOR if ar < MAX_ASPECT_RATIO else 0.
        else:
            newDelta = 1 - self.min_box_side / float(box_w)
            ar = float(box_h) / (box_w - newDelta * box_w)
            if ar > MAX_ASPECT_RATIO:
                newDelta = 0.0

        change_w = newDelta * box_w / 2.0
        newbox[0], newbox[2] = newbox[0] + change_w - 0.5, newbox[2] - change_w + 0.5

        return newbox


    def aspect_ratio_down_s(self):
        """ decrease aspect ratio, reduce box_h, keep box_w
        """
        newbox = np.copy(self.current_window)
        box_w, box_h = self._get_box_wh(newbox)
        change_w, change_h = self._get_change(STEP_FACTOR, box_w, box_h)
        if box_h - change_h >= self.min_box_side:
            ar = (box_h - change_h) / float(box_w)
            newDelta = STEP_FACTOR if ar < MAX_ASPECT_RATIO else 0.
        else:
            newDelta = 1. - self.min_box_side / float(box_h)
            ar = (box_h - newDelta * box_h) / float(box_w)
            if ar < MIN_ASPECT_RATIO:
                newDelta = 0.0

        change_h = newDelta * box_h / 2.0
        newbox[1], newbox[3] = newbox[1] + change_h - 0.5, newbox[3] - change_h + 0.5

        return newbox

    def adjust(self):
        """Check whether the new box is out of boundaries
        """
        '''check if x coordinate of the top left corner is out of boundary'''
        if self.current_window[0] < 0:
            step = - self.current_window[0]
            if self.current_window[2] + step < self.img_w:
                self.current_window[0], self.current_window[2] = self.current_window[0] + step, \
                                                                 self.current_window[2] + step
            else:
                self.current_window[0], self.current_window[2] = 0, self.img_w - 1

        '''check if y coordinate of the top left corner is out of boundary'''
        if self.current_window[1] < 0:
            step = - self.current_window[1]
            if self.current_window[3] + step < self.img_h:
                self.current_window[1], self.current_window[3] = self.current_window[1] + step, \
                                                                 self.current_window[3] + step
            else:
                self.current_window[1], self.current_window[3] = 0, self.img_h - 1

        '''check if x coordinate of the bottom right corner is out of boundary'''
        if self.current_window[2] >= self.img_w:
            step = self.current_window[2] - self.img_w + 1
            if self.current_window[0] - step >= 0:
                self.current_window[0], self.current_window[2] = self.current_window[0] - step, \
                                                                 self.current_window[2] - step
            else:
                self.current_window[0], self.current_window[2] = 0, self.img_w - 1

        '''check if y coordinate of the bottome right corner is out of boundary'''
        if self.current_window[3] >= self.img_h:
            step = self.current_window[3] - self.img_h + 1
            if self.current_window[1] - step >= 0:
                self.current_window[1], self.current_window[3] = self.current_window[1] - step, \
                                                                 self.current_window[3] - step
            else:
                self.current_window[1], self.current_window[3] = 0, self.img_h - 1

        box_w, box_h = self._get_box_wh(self.current_window)
        '''check if box_w is smaller than min_box_side'''
        if box_w < self.min_box_side:
            # TODO check if x2+min_box_size larger than img_w
            if self.current_window[0] + self.min_box_side < self.img_w:
                self.current_window[2] = self.current_window[0] + self.min_box_side - 1
            else:
                self.current_window[0] = self.current_window[2] - self.min_box_side + 1

        '''check if box_h is smaller than min_box_side'''
        if box_h < self.min_box_side:
            # TODO check if y2+min_box_size larger than img_h
            if self.current_window[1] + self.min_box_side < self.img_w:
                self.current_window[3] = self.current_window[1] + self.min_box_side - 1
            else:
                self.current_window[1] = self.current_window[3] - self.min_box_side + 1


    def _calculate_newDelta(self, boxW, boxH, widthChange, heightChange):
        if boxW - widthChange >= self.min_box_side:
            if boxH - heightChange >= self.min_box_side:
                newDelta = STEP_FACTOR
            else:
                newDelta = 1. - self.min_box_side / boxH
        else:
            newDelta = 1. - self.min_box_side / boxW
            if boxH - newDelta * boxH < self.min_box_side:
                newDelta = 1. - self.min_box_side / boxH

        return  newDelta

    def _get_box_wh(self, box):
        h = box[3] - box[1] + 1
        w = box[2] - box[0] + 1
        return w, h

    def _get_change(self, delta, box_w, box_h):
        change_w = delta * box_w
        change_h = delta * box_h
        return change_w, change_h

    def _get_scale_change(self, newbox):
        box_w, box_h = self._get_box_wh(newbox)
        change_w, change_h = self._get_change(STEP_FACTOR, box_w, box_h)
        newDelta = self._calculate_newDelta(box_w, box_h, change_w, change_h)
        change_w, change_h = self._get_change(newDelta, box_w, box_h)
        return change_w, change_h

if __name__ == '__main__':
    img_w, img_h = 28, 28
    batch_size = 3
    min_box_side = 25
    action = torch.tensor([[3], [0], [10]])
    actor = Actor((img_w, img_h), batch_size, min_box_side)
    pred_box = actor.take_action(action)
    print('pred_box ', pred_box)

