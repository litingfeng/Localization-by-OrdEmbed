"""
TODO top right coordinates -1
https://github.com/otoofim/ObjLocalisation/blob/master/lib/Agent.py
Created on 10/8/2020 9:24 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
import numpy as np
import torch
import torch.nn.functional as F

# Actions defined as in Tree-RL object detection
# Scaling
SCALE_TL = 0 # shrink toward top left
SCALE_TR = 1 # shrink toward top right
SCALE_BL = 2 # shrink toward bottom left
SCALE_BR = 3 # shrink toward bottom right
SCALE_DOWN = 4
PLACE_LANDMARK = 5
# Translation
MOVE_LEFT         = 6
MOVE_RIGHT          = 7
MOVE_UP            = 8
MOVE_DOWN          = 9
ASPECT_RATIO_UP    = 10  # enlarge H, keep W
ASPECT_RATIO_DOWN  = 11 # enlarge W, heep H
ASPECT_RATIO_UP_S  = 12 # reduce W, keep H
ASPECT_RATIO_DOWN_S  = 13 # reduce H, keep W

# Transformation coefficients
STEP_FACTOR = 0.2 #default=0.2
MAX_ASPECT_RATIO = 3.00 # for larger digits
MIN_ASPECT_RATIO = 0.3
# MAX_ASPECT_RATIO = 4.00
# MIN_ASPECT_RATIO = 0.2

class Actor:
    def __init__(self, images, targets, min_box_side=50):
        self.images = images
        self.targets = targets
        self.ious = torch.zeros(images.shape[0]) #TODO initialize iou
        self.min_box_side = min_box_side
        self.image_w, self.image_h = self.images.shape[3], self.images.shape[2]
        self.agent_windows = np.repeat(np.array([[0, 0, self.image_w-1, self.image_h-1]]),
                                      images.shape[0], axis=0)

    def takeaction(self, actions):
        """
        This function performs actions and computes the new window
        Args:
        action: Action that is going to be taken
        Returns:
        Reward corresponding to the taken action
        """
        rewards, new_boxes = [], []
        for i, action in enumerate(actions):
            termination = False
            newbox = np.array([0., 0., 0., 0.]) # [x1, y1, x2, y2]
            self.agent_window = self.agent_windows[i]
            if action == MOVE_RIGHT:
                newbox = self.MoveRight()
            elif action == MOVE_DOWN:
                newbox = self.MoveDown()
            elif action == ASPECT_RATIO_UP:
                newbox = self.aspectRatioUp()
            elif action == MOVE_LEFT:
                newbox = self.MoveLeft()
            elif action == MOVE_UP:
                newbox = self.MoveUp()
            elif action == SCALE_DOWN:
                newbox = self.scaleDown()
            elif action == ASPECT_RATIO_DOWN:
                newbox = self.aspectRatioDown()
            elif action == SCALE_TL:
                newbox = self.scaleTL()
            elif action == SCALE_TR:
                newbox = self.scaleTR()
            elif action == SCALE_BL:
                newbox = self.scaleBL()
            elif action == SCALE_BR:
                newbox = self.scaleBR()
            elif action == ASPECT_RATIO_UP_S:
                newbox = self.aspectRatioUpS()
            elif action == ASPECT_RATIO_DOWN_S:
                newbox = self.aspectRatioDownS()
            elif action == PLACE_LANDMARK:
                newbox = self.placeLandmark()
                termination = True

            # Storing new window
            self.agent_window = newbox
            # Checking whether the new window is out of boundaries
            self.adjustAndClip()

            self.agent_windows[i] = self.agent_window

            # computing the reward and IoU corresponding to the taken action
            r, new_iou = self.ComputingReward(self.agent_window, self.targets[i].numpy(),
                                              self.ious[i], termination)
            self.ious[i] = new_iou

            new_boxes.append(torch.from_numpy(self.agent_window))
            rewards.append(r)

        newboxes = torch.stack(new_boxes).float()
        rewards = torch.from_numpy(np.array(rewards))
        #ious = self.ious.clone()

        return rewards, newboxes

    def ComputingReward(self, agent_window, target, old_iou, termination=False):
        """
        Going over the all bounding boxes to compute the reward for a given action.
        Args:
        agent_window: Current agent window
        termination: Is this termination action?
        Returns:
        Reward and IoU coresponding to the agent window
        """
        # Computing IoU between agent window and ground truth
        new_iou = self.intersectionOverUnion(agent_window, target)
        reward = self.ReturnReward(new_iou, old_iou, termination)

        # if termination:
        #     new_iou = 0
        return reward, new_iou


    def ReturnReward(self, new_iou, old_iou, termination):
        """
        Computing reward regarding new window
        Args:
        new_iou: new IoU corresponding to the recent action
        termination: Is this termination action?
        Returns:
        Reward corresponding to the action
        """
        reward = 0.
        # If new IoU is bigger than the last IoU the agent will be recieved positive reward
        if new_iou - old_iou > 0:
            reward = 1.
        elif new_iou - old_iou == 0:
            reward = 0.
        else:
            reward = -1

        # If the action is the trigger then the new IoU will compare to the threshold 0.5
        if termination:
            if (new_iou > 0.5):
                reward = 3.
            else:
                reward = -3.
        return reward

    def intersectionOverUnion(self, boxA, boxB):
        """
        Computing IoU
        Args:
        boxA: First box
        boxB: Second box
        Returns:
        IoU for the given boxes
        """

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground_truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def MoveRight(self):
            """
            Action moving right
            Returns:
            New window
            """

            newbox = np.copy(self.agent_window)
            boxW = newbox[2] - newbox[0]
            step = STEP_FACTOR * boxW
            # This action preserves box width and height
            if newbox[2] + step < self.image_w:
                newbox[0] += step
                newbox[2] += step

            else:
                newbox[0] = self.image_w - boxW - 1
                newbox[2] = self.image_w - 1

            return newbox


    def MoveDown(self):
        """
        Action moving down
        Returns:
        New window
        """
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        step = STEP_FACTOR * boxH
        # This action preserves box width and height
        if newbox[3] + step < self.image_h:
            newbox[1] += step
            newbox[3] += step
        else:
            newbox[1] = self.image_h - boxH - 1
            newbox[3] = self.image_h - 1

        return newbox


    def scaleUp(self):
        """
        Action scaling up
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        boxH = newbox[3] - newbox[1]

        # This action preserves aspect ratio
        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        if boxW + widthChange < self.image_w:
            if boxH + heightChange < self.image_h:
                newDelta = STEP_FACTOR
            else:
                newDelta = self.image_h / boxH - 1
        else:
            newDelta = self.image_w / boxW - 1
            if boxH + (newDelta * boxH) >= self.image_h:
                newDelta = self.image_h / boxH - 1

        widthChange = newDelta * boxW / 2.0
        heightChange = newDelta * boxH / 2.0
        newbox[0] -= widthChange
        newbox[1] -= heightChange
        newbox[2] += widthChange
        newbox[3] += heightChange

        return newbox

    def scaleTL(self):
        """
        shrink toward top left
        :return:
        """
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        newDelta = self._calulate_newDelta(boxW, boxH, widthChange,
                                           heightChange)
        widthChange = newDelta * boxW
        heightChange = newDelta * boxH
        newbox[2] -= widthChange
        newbox[3] -= heightChange

        return newbox

    def scaleTR(self):
        """
        shrink toward top right
        :return:
        """
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        newDelta = self._calulate_newDelta(boxW, boxH, widthChange,
                                           heightChange)
        widthChange = newDelta * boxW
        heightChange = newDelta * boxH
        newbox[0] += widthChange
        newbox[3] -= heightChange

        return newbox

    def scaleBL(self):
        """
        shrink toward bottom left
        :return:
        """
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        newDelta = self._calulate_newDelta(boxW, boxH, widthChange,
                                           heightChange)
        widthChange = newDelta * boxW
        heightChange = newDelta * boxH
        newbox[1] += heightChange
        newbox[2] -= widthChange

        return newbox

    def scaleBR(self):
        """
        shrink toward bottom left
        :return:
        """
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        newDelta = self._calulate_newDelta(boxW, boxH, widthChange,
                                           heightChange)
        widthChange = newDelta * boxW
        heightChange = newDelta * boxH
        newbox[0] += widthChange
        newbox[1] += heightChange

        return newbox

    def aspectRatioUp(self):
        """
        Action increasing aspect ratio
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        # This action preserves width
        heightChange = STEP_FACTOR * boxH

        if boxH + heightChange < self.image_h:
            ar = (boxH + heightChange) / boxW
            if ar < MAX_ASPECT_RATIO:
                newDelta = STEP_FACTOR
            else:
                newDelta = 0.0
        else:
            newDelta = self.image_h / boxH - 1
            ar = (boxH + newDelta * boxH) / boxW
            if ar > MAX_ASPECT_RATIO:
                newDelta = 0.0

        heightChange = newDelta * boxH / 2.0
        newbox[1] -= heightChange
        newbox[3] += heightChange

        return newbox

    def aspectRatioUpS(self):
        """
        Action increasing aspect ratio, reduce W, keep H
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        # This action preserves height
        widthChange = STEP_FACTOR * boxW

        if boxW - widthChange >= self.min_box_side:
            ar = boxH / (boxW - widthChange)
            if ar <= MAX_ASPECT_RATIO:
                newDelta = STEP_FACTOR
            else:
                newDelta = 0.0
        else:
            newDelta = 1 - self.min_box_side / boxW
            ar = boxH / (boxW - newDelta * boxW) # boxH/min
            if ar > MAX_ASPECT_RATIO:
                newDelta = 0.0

        widthChange = newDelta * boxW / 2.0
        newbox[0] += widthChange
        newbox[2] -= widthChange

        return newbox



    def MoveLeft(self):
        """
        Action moving left
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        step = STEP_FACTOR * boxW

        # This action preserves box width and height
        if newbox[0] - step >= 0:
            newbox[0] -= step
            newbox[2] -= step
        else:
            newbox[0] = 0
            newbox[2] = boxW

        return newbox


    def MoveUp(self):
        """
        Action moving up
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)

        boxH = newbox[3] - newbox[1]

        step = STEP_FACTOR * boxH
        # This action preserves box width and height
        if newbox[1] - step >= 0:
            newbox[1] -= step
            newbox[3] -= step
        else:
            newbox[1] = 0
            newbox[3] = boxH

        return newbox


    def scaleDown(self):
        """
        Action moving down, bug fixed
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)

        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        # This action preserves aspect ratio
        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        newDelta = self._calulate_newDelta(boxW, boxH, widthChange,
                                           heightChange)

        widthChange = newDelta * boxW / 2.0
        heightChange = newDelta * boxH / 2.0
        newbox[0] += widthChange
        newbox[1] += heightChange
        newbox[2] -= widthChange
        newbox[3] -= heightChange

        return newbox


    def splitHorizontal(self):
        """
        Action horizontal splitting
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        if boxW > self.min_box_side:
            half = boxW / 2.0
            newbox[2] -= half
        return newbox


    def splitVertical(self):
        """
        Action vertical splitting
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        if boxH > self.min_box_side:
            half = boxH / 2.0
            newbox[3] -= half
        return newbox


    def aspectRatioDown(self):
        """
        Action decreasing aspect ratio
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)

        boxW = newbox[2] - newbox[0]
        boxH = newbox[3] - newbox[1]

        # This action preserves height
        widthChange = STEP_FACTOR * boxW
        if boxW + widthChange < self.image_w:
            ar = boxH / (boxW + widthChange)
            if ar >= MIN_ASPECT_RATIO:
                newDelta = STEP_FACTOR
            else:
                newDelta = 0.0
        else:
            newDelta = self.image_w / boxW - 1
            ar = boxH / (boxW + newDelta * boxW)
            if ar < MIN_ASPECT_RATIO:
                newDelta = 0.0
        widthChange = newDelta * boxW / 2.0
        newbox[0] -= widthChange
        newbox[2] += widthChange

        return newbox

    def aspectRatioDownS(self):
        """
        Action decreasing aspect ratio, reduce h, keep w
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        # This action preserves width
        heightChange = STEP_FACTOR * boxH

        if boxH - heightChange >= self.min_box_side:
            ar = (boxH - heightChange) / boxW
            if ar > MIN_ASPECT_RATIO:
                newDelta = STEP_FACTOR
            else:
                newDelta = 0.0
        else:
            newDelta = 1. - self.min_box_side / boxH
            ar = (boxH - newDelta * boxH) / boxW
            if ar < MIN_ASPECT_RATIO:
                newDelta = 0.0

        heightChange = newDelta * boxH / 2.0
        newbox[1] += heightChange
        newbox[3] -= heightChange

        return newbox

    def placeLandmark(self):
        """
        Termination action. This action returns the last window without any changes however for visualization purposes a black cross sign is put on the image to detemine search termination
        Returns:
        New window
        """

        newbox = np.copy(self.agent_window)

        # h = int((newbox[3] - newbox[1]) / 2)
        # h_l = int(h / 5)
        # w = int((newbox[2] - newbox[0]) / 2)
        # w_l = int(w / 5)
        #
        # self.image_playground[newbox[1] + h - h_l:newbox[1] + h + h_l, newbox[0]:newbox[2]] = 0
        # self.image_playground[newbox[1]:newbox[3], newbox[0] + w - w_l:newbox[0] + w + w_l] = 0

        return newbox


    def adjustAndClip(self):
        """
        Cheching whether the new window is out of boundaries
        """

        # Cheching if x coordinate of the top left corner is out of bound
        if self.agent_window[0] < 0:
            #print('self.agent ', self.agent_window)
            step = -self.agent_window[0]
            if self.agent_window[2] + step < self.image_w:
                self.agent_window[0] += step
                self.agent_window[2] += step
            else:
                self.agent_window[0] = 0
                self.agent_window[2] = self.image_w - 1
            #print('after self.agent ', self.agent_window)

        # Cheching if y coordinate of the top left corner is out of bound
        if self.agent_window[1] < 0:
            #print('self.agent ', self.agent_window)
            step = -self.agent_window[1]
            if self.agent_window[3] + step < self.image_h:
                self.agent_window[1] += step
                self.agent_window[3] += step
            else:
                self.agent_window[1] = 0
                self.agent_window[3] = self.image_h - 1
            #print('after self.agent ', self.agent_window)
            #exit()

        # Cheching if x coordinate of the bottom right corner is out of bound
        if self.agent_window[2] > self.image_w - 1:
            step = self.agent_window[2] - self.image_w + 1
            if self.agent_window[0] - step >= 0:
                self.agent_window[0] -= step
                self.agent_window[2] -= step
            else:
                self.agent_window[0] = 0
                self.agent_window[2] = self.image_w - 1

        # Cheching if y coordinate of the bottom right corner is out of bound
        if self.agent_window[3] > self.image_h - 1:
            step = self.agent_window[3] - self.image_h + 1
            if self.agent_window[1] - step >= 0:
                self.agent_window[1] -= step
                self.agent_window[3] -= step
            else:
                self.agent_window[1] = 0
                self.agent_window[3] = self.image_h - 1

        if self.agent_window[2] - self.agent_window[0] < self.min_box_side:
        #if self.agent_window[0] == self.agent_window[2]:
            if self.agent_window[0] + self.min_box_side <= self.image_w:
                self.agent_window[2] = self.agent_window[0] + self.min_box_side - 1
            else:
                self.agent_window[0] = self.agent_window[2] - self.min_box_side + 1
            #print('2after self.agent ', self.agent_window)

        if self.agent_window[3] - self.agent_window[1] < self.min_box_side:
        #if self.agent_window[1] == self.agent_window[3]:
            if self.agent_window[1] + self.min_box_side <= self.image_h:
                self.agent_window[3] = self.agent_window[1] + self.min_box_side - 1
            else:
                self.agent_window[1] = self.agent_window[3] - self.min_box_side + 1
            #print('2after self.agent ', self.agent_window)

    def _calulate_newDelta(self, boxW, boxH, widthChange, heightChange):

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

if __name__ == '__main__':
    images = torch.rand(3, 1, 56, 28)
    actor = Actor(images, min_box_side=25)
    action = torch.tensor([[3]])
    _ = actor.takeaction(action)
    exit()
    masks = actor.get_invalid_actions()
    print('masks ', masks)
    logits = torch.rand(3, 9)
    print('logits ', logits)
    adj_logits = logits * masks
    print(adj_logits)
    print(torch.where(masks, logits, torch.tensor(-1e+8)))
