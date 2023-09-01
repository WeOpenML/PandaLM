import json
from collections import defaultdict

# Paths to the ground truth and model output JSON files
ground_truth_path = 'testset-v1.json'

model_output_path = 'gpt-3.5-turbo-testset-v1.json'
# Accuracy: 0.7107
# Macro-Averaged Precision: 0.5879
# Macro-Averaged Recall: 0.5736
# Macro-Averaged F1 Score: 0.5755

#model_output_path = 'pandalm-7b-testset-v1.json'
# Accuracy: 0.6677
# Macro-Averaged Precision: 0.5738
# Macro-Averaged Recall: 0.5750
# Macro-Averaged F1 Score: 0.5743

def get_majority_vote(*votes):
    vote_count = defaultdict(int)
    for vote in votes:
        vote_count[vote] += 1
    return max(vote_count, key=vote_count.get)

def compute_metrics(ground_truth_path, model_output_path):
    with open(ground_truth_path, 'r') as gt_file, open(model_output_path, 'r') as mo_file:
        ground_truths = json.load(gt_file)
        model_outputs = json.load(mo_file)

        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for gt, mo in zip(ground_truths, model_outputs):
            majority_vote = get_majority_vote(gt['annotator1'], gt['annotator2'], gt['annotator3'])
            if 'gpt' in model_output_path:
                model_prediction = mo['gpt_result']
            elif 'panda' in model_output_path:
                model_prediction = mo['pandalm_result']
            if model_prediction == "Tie" or model_prediction == "tie" or model_prediction =="garbage":
                model_prediction = 0
            else:
                model_prediction = int(model_prediction)

            confusion_matrix[majority_vote][model_prediction] += 1

        # Accuracy calculation
        possible_classes = [0, 1, 2]
        total_true_positives = sum(confusion_matrix[i][i] for i in possible_classes)
        total_instances = sum(sum(confusion_matrix[i].values()) for i in possible_classes)
        accuracy = total_true_positives / total_instances

        metrics = {}
        
        for cls in possible_classes:
            TP = confusion_matrix[cls][cls]
            FP = sum(confusion_matrix[x][cls] for x in possible_classes) - TP
            FN = sum(confusion_matrix[cls][x] for x in possible_classes) - TP
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}

        macro_avg = {}
        for metric in ['precision', 'recall', 'f1']:
            macro_avg[metric] = sum(metrics[cls][metric] for cls in possible_classes) / 3

        return accuracy, macro_avg


accuracy, macro_avg = compute_metrics(ground_truth_path, model_output_path)
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-Averaged Precision: {macro_avg['precision']:.4f}")
print(f"Macro-Averaged Recall: {macro_avg['recall']:.4f}")
print(f"Macro-Averaged F1 Score: {macro_avg['f1']:.4f}")
