import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
# forward is implemented in ultils.py in the same package
from .ultils import forward

class Evaluator(object):
    def __init__(self, model):
        """Simple evaluator wrapper.

        model: a PyTorch model that returns a dict with key 'clipwise_output'.
        """
        self.model = model

    def evaluate(self, data_loader):
        """
        Run model on `data_loader` and compute metrics.

        Returns a dict with keys: 'average_precision', 'accuracy', 'auc', 'message'.
        """

        # Forward
        output_dict = forward(
            model=self.model,
            generator=data_loader,
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        try:
            # Caculate mAP (Average Precision)
            average_precision = metrics.average_precision_score(
                target, clipwise_output, average=None)
        except ValueError as e:
            print(f"Lỗi khi tính mAP (có thể do các lớp bị thiếu trong batch): {e}")
            average_precision = np.zeros(target.shape[1])

        try:
            # Calculate AUC
            auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        except ValueError as e:
            print(f"Error calculating AUC (possibly due to missing classes in batch): {e}")
            auc = np.zeros(target.shape[1])

        # Calculate Accuracy
        target_acc = np.argmax(target, axis=1)
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        acc = accuracy_score(target_acc, clipwise_output_acc)

        # Create detailed report (message)
        labels = ['None', 'Strong', 'Medium', 'Weak'] # Based on your one-hot encoding order
        message = "\nMetrics per class (None, Strong, Medium, Weak):\n"
        message += f"  mAP: {np.round(average_precision, 3)}\n"
        message += f"  AUC: {np.round(auc, 3)}\n"

        statistics = {'average_precision': average_precision, 'accuracy': acc, 'auc': auc, 'message': message}

        return statistics