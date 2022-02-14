import pickle
import os
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, path="..\\ML_data\\model.pt"):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.path = path
        self.best_loss = None
        self.early_stop = False
        self.current_best = False
    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            self.current_best = True
            pickle.dump(model, open(os.path.join(self.path), "wb"))
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
            self.current_best = False