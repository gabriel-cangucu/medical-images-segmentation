import torch


class RunningMetrics(object):

    def __init__(self, n_classes, bf1_threshold=None):
        self.n_classes = n_classes
        self.bf1_threshold = bf1_threshold

        self.confusion_matrix = torch.zeros((n_classes, n_classes))
        self.bf1_matrix = [0.] if self.bf1_threshold is None else list()
    

    def __compute_matrix(self, pred_label, true_label, n_classes):
        mask = (true_label >= 0) & (true_label < n_classes)
        hist = torch.bincount(
            n_classes*true_label[mask].to(int) + pred_label[mask], minlength=n_classes*n_classes
        ).reshape(n_classes, n_classes)
        
        return hist.cpu()
    

    def __find_class_contours(self, matrix, label):
        label_matrix = (matrix == label).int()

        gt_tb  = label_matrix[1:, :] - label_matrix[:-1, :]
        gt_lr  = label_matrix[:, 1:] - label_matrix[:, :-1]
        gt_tb  = torch.nn.functional.pad(input=gt_tb, pad=[0, 0, 0, 1], mode='constant', value=0) != 0 
        gt_lr  = torch.nn.functional.pad(input=gt_lr, pad=[0, 1, 0, 0], mode='constant', value=0) != 0
        gt_idx = torch.nonzero((gt_lr + gt_tb) == 1, as_tuple=False)

        return gt_idx


    def __precision_recall(self, vector_a, vector_b, threshold=2):
        '''
            For precision, 'vector_a' = ground truth & 'vector_b' = predictions
            For precision, 'vector_a' = predictions & 'vector_b' = ground truth
        '''

        # Constrain long arrays when their size differ significantly
        upper_bound = max([len(vector_a), len(vector_b)])
        lower_bound = min([len(vector_a), len(vector_b)])
        bound = upper_bound if (upper_bound / lower_bound <= 2.) else lower_bound

        # Shrinking vectors 
        vector_a = vector_a[:bound].float()
        vector_b = vector_b[:bound].float()

        # Efficient implementation of the Euclidean Distance
        distance  = torch.cdist(vector_a, vector_b, p=2)
        top_count = torch.any(distance < threshold, dim=0).sum()

        try:
            precision_recall = top_count / len(vector_b)
        except ZeroDivisionError:
            precision_recall = 0
        return precision_recall, top_count, len(vector_b)


    def __compute_boundary(self, pred_label, true_label):
        device_idx = true_label.get_device()
        device = device_idx if device_idx >= 0 else 'cpu'

        bf1_scores = torch.zeros(self.n_classes, device=device)

        for label in range(self.n_classes):
            # Removing len=1 axes from matrices
            preds, trues = pred_label.squeeze(), true_label.squeeze()

            # Find matrix indices storing boundaries
            contour_pr = self.__find_class_contours(preds, label)
            contour_tr = self.__find_class_contours(trues, label)

            # Compute BF1 Score
            if len(contour_pr) and len(contour_tr):
                # Compute Precision and Recall
                precis, pre_num, pre_den = self.__precision_recall(contour_tr, contour_pr, self.bf1_threshold)
                recall, rec_num, rec_den = self.__precision_recall(contour_pr, contour_tr, self.bf1_threshold)

                bf1_scores[label] += (2*recall*precis / (recall + precis)) if (recall + precis) > 0 else 0.
            else:
                bf1_scores[label] += 0

        return bf1_scores
    

    def update(self, images, targets):
        pred_labels = images.detach().max(dim=1)[1]
        true_labels = targets.detach()

        for p_label, t_label in zip(pred_labels, true_labels):
            self.confusion_matrix += self.__compute_matrix(p_label.flatten(), t_label.flatten(), self.n_classes)

            if self.bf1_threshold is not None:
                self.bf1_matrix.append(self.__compute_boundary(p_label, t_label)) 


    def get_scores(self):
        '''
        Computes and returns the following metrics:

            - Pixel Accuracy
            - Class Accuracy
            - Mean Class Accuracy
            - Mean Intersection Over Union (mIoU)
            - Frequency Weighted IoU
            - Confusion Matrix
            - BF1 Score
        '''

        hist = self.confusion_matrix

        pixel_accuracy = torch.nan_to_num(torch.diag(hist).sum() / hist.sum(), nan=0.0)
        class_accuracy = torch.nan_to_num(torch.diag(hist) / hist.sum(dim=1), nan=0.0)
        mean_class_accuracy = torch.nanmean(class_accuracy)

        iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        mean_iou = torch.nanmean(iou)

        frequency = hist.sum(dim=1) / hist.sum() # fraction of the pixels that come from each class
        frequency_weighted_iou = (frequency[frequency > 0] * iou[frequency > 0]).sum()

        bf1_matrix = torch.mean(torch.stack(self.bf1_matrix))

        return {
            'pixel_accuracy'        : pixel_accuracy.item(),
            'class_accuracy'        : class_accuracy.tolist(),
            'mean_class_accuracy'   : mean_class_accuracy.item(),
            'mean_iou'              : mean_iou.item(),
            'frequency_weighted_iou': frequency_weighted_iou.item(),
            'confusion_matrix'      : self.confusion_matrix.tolist(),
            'bf1_score'             : bf1_matrix.item()
        }
    

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))
        self.bf1_matrix = [0] if self.bf1_threshold is None else list()


class EarlyStopper:

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            
            if self.counter >= self.patience:
                return True
            
        return False

