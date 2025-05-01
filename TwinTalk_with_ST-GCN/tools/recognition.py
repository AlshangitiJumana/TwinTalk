# st-gcn/tools/recognition.py

import yaml
import torch
import numpy as np
import os
import sys
import importlib

from torchlight import IO, import_class
from sklearn.metrics import accuracy_score, recall_score, f1_score


class Processor:
    """
    Processor for ST-GCN models.
    Safely loads pretrained weights, replaces input/output layers if necessary,
    freezes early layers, and handles training and testing modes.
    """

    def __init__(self, arg):
        self.arg = arg
        self.training_metrics = []  # To store metrics per epoch

        # --- Load configuration ---
        with open(self.arg.config, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # --- Setup working directory ---
        if not self.arg.work_dir:
            self.arg.work_dir = self.config.get('work_dir', './work_dir/tmp')
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        # --- Setup logging ---
        self.print_log = print if getattr(self.arg, 'print_log', True) else lambda *a, **k: None

        # --- Load model ---
        Model = import_class(self.config['model'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Model(**self.config.get('model_args', {})).to(self.device)

        # --- Load pretrained weights safely ---
        weights_path = self.config.get('weights')
        if weights_path and os.path.isfile(weights_path):
            self._load_pretrained_weights(weights_path)

        # --- Setup data loader ---
        Feeder = import_class(self.config['train_feeder'])
        self.data_loader = torch.utils.data.DataLoader(
            dataset=Feeder(**self.config.get('train_feeder_args', {})),
            batch_size=self.config.get('train_batch_size', 64),
            shuffle=True if self.arg.phase == 'train' else False,
            num_workers=0
        )

        self.optimizer = None

    def _load_pretrained_weights(self, weights_path):
        """
        Load pretrained weights safely, replace input/output layers if needed,
        and freeze all layers except important parts for fine-tuning.
        """
        self.print_log(f"🚀 Loading pretrained weights from {weights_path}")

        pretrained_dict = torch.load(weights_path, map_location=self.device)
        model_dict = self.model.state_dict()

        # Filter matching keys (name and shape)
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)

        self.print_log(f"✅ {len(filtered_dict)} layers loaded successfully.")

        # --- Replace input layer if needed ---
        expected_in_channels = self.config['model_args']['in_channels']
        num_point = self.config['model_args']['num_point']

        if hasattr(self.model, 'data_bn'):
            real_in_channels = self.model.data_bn.num_features // num_point
            if expected_in_channels != real_in_channels:
                self.print_log(f"⚡ Replacing input layer: {real_in_channels} ➔ {expected_in_channels} channels")
                self.model.data_bn = torch.nn.BatchNorm1d(expected_in_channels * num_point)
        else:
            self.print_log(f"⚠️ Warning: Model has no 'data_bn'. Skipping input layer adjustment.")

        # --- Replace output layer if needed ---
        expected_classes = self.config['model_args']['num_class']
        if hasattr(self.model, 'fcn'):
            old_out_features = self.model.fcn.out_channels
            if old_out_features != expected_classes:
                self.print_log(f"⚡ Replacing output layer: {old_out_features} ➔ {expected_classes} classes")
                self.model.fcn = torch.nn.Conv2d(256, expected_classes, 1)
        else:
            self.print_log(f"⚠️ Warning: Model has no 'fcn'. Skipping output layer adjustment.")

        # --- Freeze all layers initially ---
        for param in self.model.parameters():
            param.requires_grad = False

        # --- Unfreeze important parts for fine-tuning ---
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ['fcn', 'st_gcn_networks.6', 'st_gcn_networks.5', 'data_bn']):
                param.requires_grad = True

        self.print_log("❄️ Freezing applied. Only last GCN layers + input/output layers are trainable.")

    def start(self):
        """Start either training or testing based on the phase."""
        self.print_log('Processor started')
        if self.arg.phase == 'train':
            self.train()
        elif self.arg.phase == 'test':
            self.test()

    def train(self):
        """Training loop."""
        self.model.train()
        self.optimizer_initialized = False
        num_epoch = self.config.get('num_epoch', 50)

        self.print_log(f"🧠 Training for {num_epoch} epochs...")

        for epoch in range(num_epoch):
            print(f"\n🌀 Epoch {epoch + 1}/{num_epoch}")

            all_preds = []
            all_labels = []

            for batch_idx, (data, label) in enumerate(self.data_loader):
                data = data.to(self.device).float()
                label = label.to(self.device).long()

                # Add velocity channel
                velocity = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
                data = torch.cat((data[:, :, :-1, :, :], velocity), dim=1)
                data = data.squeeze(-1)

                output = self.model(data)
                preds = torch.argmax(output, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                if not self.optimizer_initialized:
                    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
                    self.optimizer_initialized = True

                loss = torch.nn.CrossEntropyLoss()(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"  🔁 Batch {batch_idx + 1}/{len(self.data_loader)} | Loss: {loss.item():.4f}")

            # Epoch metrics
            acc = accuracy_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            self.training_metrics.append({
                'epoch': epoch + 1,
                'accuracy': acc,
                'recall': recall,
                'f1': f1
            })

            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   ✅ Accuracy: {acc:.4f}")
            print(f"   🔁 Recall:   {recall:.4f}")
            print(f"   ⭐ F1 Score: {f1:.4f}")

    def test(self):
        """Testing loop."""
        self.model.eval()
        correct = 0
        total = 0

        print("🧪 Testing the model...")

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.data_loader):
                data = data.to(self.device).float()
                label = label.to(self.device).long()

                # Add velocity
                velocity = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
                data = torch.cat((data[:, :, :-1, :, :], velocity), dim=1)
                data = data.squeeze(-1)

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = correct / total
        print(f"\n🎯 Test Accuracy: {accuracy:.4f}")
