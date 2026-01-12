"""Training and evaluation utilities."""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class Trainer:
    """Training and evaluation manager."""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3, 
                 optimizer_type='adam', scheduler_type='cosine'):
        """
        Args:
            model: PyTorch model
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate
            optimizer_type: 'adam' or 'sgd'
            scheduler_type: 'cosine' or 'step'
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if optimizer_type == 'adam':
            self.optimizer = Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=1e-4)
        elif optimizer_type == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=learning_rate, 
                                momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Scheduler
        self.scheduler = None
        self.scheduler_type = scheduler_type
        
        # Metrics
        self.train_history = {
            'loss': [],
            'accuracy': []
        }
        self.val_history = {
            'loss': [],
            'accuracy': []
        }
    
    def set_scheduler(self, num_epochs):
        """Set learning rate scheduler."""
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif self.scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def test(self, test_loader):
        """Test on test set with detailed metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', 
                                   zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', 
                             zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Confusion matrix and classification report
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(self, train_loader, val_loader, num_epochs=100, checkpoint_dir='./checkpoints'):
        """Full training loop."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        self.set_scheduler(num_epochs)
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(checkpoint_path / 'best_model.pth')
                print(f"Best model saved! Val Acc: {best_val_acc:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_path / f'epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def save_history(self, path):
        """Save training history to JSON."""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)


def get_optimizer_and_scheduler(model, num_epochs, learning_rate=1e-3,
                               optimizer_type='adam', scheduler_type='cosine'):
    """Create optimizer and scheduler.
    
    Args:
        model: PyTorch model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        optimizer_type: 'adam' or 'sgd'
        scheduler_type: 'cosine' or 'step'
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, 
                       weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return optimizer, scheduler
