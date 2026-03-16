import json
import numpy as np

from .config import CDConfig


class CDECVMetrics:
    def __init__(self, cfg: CDConfig):
        self.cfg = cfg
        self.total = self.correct = self.abstentions = 0
        self.tp = self.pp = self.gp = 0
        self.per_class_correct = {label: 0 for label in cfg.all_labels}
        self.per_class_total = {label: 0 for label in cfg.all_labels}
        self.confusion = {
            gold: {pred: 0 for pred in cfg.all_labels}
            for gold in cfg.all_labels
        }

    @staticmethod
    def norm(label):
        if not label:
            return 'NOT_ENOUGH_INFO'
        label = label.strip().upper()
        if label in ('SUPPORT', 'SUPPORTED', 'SUPPORTS'):
            return 'SUPPORT'
        if label in ('CONTRADICT', 'REFUTED', 'CONTRADICTS'):
            return 'CONTRADICT'
        return 'NOT_ENOUGH_INFO'

    def update(self, pred_label, gold_label, pred_ids, gold_ids):
        self.total += 1
        pred_label = self.norm(pred_label)
        gold_label = self.norm(gold_label)
        if pred_label == gold_label:
            self.correct += 1
            self.per_class_correct[gold_label] += 1
        if pred_label == 'NOT_ENOUGH_INFO':
            self.abstentions += 1
        self.per_class_total[gold_label] += 1
        self.confusion[gold_label][pred_label] += 1
        pred_ids = set(str(x) for x in pred_ids)
        gold_ids = set(str(x) for x in gold_ids)
        self.pp += len(pred_ids)
        self.gp += len(gold_ids)
        self.tp += len(pred_ids & gold_ids)

    def compute(self):
        precision = self.tp / self.pp if self.pp else 0
        recall = self.tp / self.gp if self.gp else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class_acc = {
            label: round(self.per_class_correct[label] / self.per_class_total[label], 4)
            if self.per_class_total[label] else 0.0
            for label in self.cfg.all_labels
        }
        return {
            'Total Claims': self.total,
            'Label Accuracy': round(self.correct / self.total, 4) if self.total else 0,
            'Abstention Rate': round(self.abstentions / self.total, 4) if self.total else 0,
            'Evidence Precision': round(precision, 4),
            'Evidence Recall': round(recall, 4),
            'Evidence F1': round(f1, 4),
            'Per-Class Accuracy': per_class_acc,
            'Macro Accuracy': round(sum(per_class_acc.values()) / len(self.cfg.all_labels), 4),
        }

    def print_summary(self, label='CD-ECV'):
        results = self.compute()
        print(f'\n=== {label} Evaluation Results ===')
        for key, value in results.items():
            if key == 'Per-Class Accuracy':
                print(f'  {key}:')
                for cls, acc in value.items():
                    print(f'    {cls:<20}: {acc:.4f}  (n={self.per_class_total[cls]})')
            else:
                print(f'  {key}: {value}')
        print('\n  Confusion Matrix (rows=gold, cols=pred):')
        print(f"  {'':22}" + ''.join(f'{pred:<22}' for pred in self.cfg.all_labels))
        for gold in self.cfg.all_labels:
            row = ''.join(f'{self.confusion[gold][pred]:<22}' for pred in self.cfg.all_labels)
            print(f'  {gold:<22}{row}')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def gold_evidence_ids(claim_rec):
    return set(str(k) for k in claim_rec.get('evidence', {}).keys())


def gold_label_from_rec(claim_rec):
    evidence = claim_rec.get('evidence', {})
    if not evidence:
        return 'NOT_ENOUGH_INFO'
    labels = [ev.get('label', '').upper() for ev_list in evidence.values() for ev in ev_list]
    if any(label in ('SUPPORT', 'SUPPORTED', 'SUPPORTS') for label in labels):
        return 'SUPPORT'
    if any(label in ('CONTRADICT', 'REFUTED', 'CONTRADICTS') for label in labels):
        return 'CONTRADICT'
    return 'NOT_ENOUGH_INFO'
