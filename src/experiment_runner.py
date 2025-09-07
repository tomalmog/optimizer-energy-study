"""
Neural Network Optimizer Energy Efficiency Study
==============================================================

This script conducts a systematic study of energy consumption and performance 
across multiple optimizers, datasets, and model architectures.

Requirements:
- torch, torchvision, transformers, datasets
- codecarbon
- pandas, numpy
- psutil (for detailed system monitoring)
- GPUtil (for GPU monitoring if available)

Installation:
pip install torch torchvision transformers datasets codecarbon pandas numpy psutil GPUtil
"""

import time, os, random, json, logging, gc, psutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Transformer models and datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertModel, DistilBertModel, RobertaModel,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import load_dataset

from codecarbon import EmissionsTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experimental configuration
CONFIG = {
    'seeds': list(range(15)),  # 15 runs for robust statistics
    'device': torch.device("mps" if torch.backends.mps.is_available() else 
                          ("cuda" if torch.cuda.is_available() else "cpu")),
    'output_dir': Path('./comprehensive_results'),
    'cache_dir': Path('./cache'),
    'batch_sizes': {'small': 32, 'medium': 64, 'large': 128},
    'early_stopping_patience': 5,  # epochs
    'max_epochs': {'vision': 50, 'nlp': 10, 'synthetic': 30},
    'validation_split': 0.2,
    'test_split': 0.1
}

# Optimizers with carefully tuned hyperparameters
OPTIMIZERS = {
    'SGD': lambda params: optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4),
    'Adam': lambda params: optim.Adam(params, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4),
    'AdamW': lambda params: optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
    'RMSprop': lambda params: optim.RMSprop(params, lr=0.001, alpha=0.99, weight_decay=1e-4),
    'Adagrad': lambda params: optim.Adagrad(params, lr=0.01, weight_decay=1e-4),
    'Adadelta': lambda params: optim.Adadelta(params, lr=1.0, rho=0.9, weight_decay=1e-4),
    'AdamaxV': lambda params: optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), weight_decay=1e-4),
    'NAdam': lambda params: optim.NAdam(params, lr=0.002, betas=(0.9, 0.999), weight_decay=1e-4),
}

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ModernCNN(nn.Module):
    """More complex CNN for CIFAR-10/100"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class TransformerClassifier(nn.Module):
    """Transformer-based text classifier"""
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2):
        super().__init__()
        if 'distilbert' in model_name:
            self.transformer = DistilBertModel.from_pretrained(model_name)
        elif 'roberta' in model_name:
            self.transformer = RobertaModel.from_pretrained(model_name)
        else:  # BERT-based
            self.transformer = BertModel.from_pretrained(model_name)
            
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_vision_datasets():
    """Prepare vision datasets with appropriate transforms"""
    datasets_info = {}
    
    # MNIST
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform_mnist)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform_mnist)
    
    datasets_info['MNIST'] = {
        'train': mnist_train, 'test': mnist_test,
        'model_class': SimpleCNN, 'num_classes': 10,
        'input_shape': (1, 28, 28), 'complexity': 'simple'
    }
    
    # CIFAR-10
    transform_cifar_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    cifar10_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_cifar_train)
    cifar10_test = datasets.CIFAR10('./data', train=False, transform=transform_cifar_test)
    
    datasets_info['CIFAR10'] = {
        'train': cifar10_train, 'test': cifar10_test,
        'model_class': lambda: ModernCNN(num_classes=10), 'num_classes': 10,
        'input_shape': (3, 32, 32), 'complexity': 'medium'
    }
    
    # CIFAR-100
    cifar100_train = datasets.CIFAR100('./data', train=True, download=True, transform=transform_cifar_train)
    cifar100_test = datasets.CIFAR100('./data', train=False, transform=transform_cifar_test)
    
    datasets_info['CIFAR100'] = {
        'train': cifar100_train, 'test': cifar100_test,
        'model_class': lambda: ModernCNN(num_classes=100), 'num_classes': 100,
        'input_shape': (3, 32, 32), 'complexity': 'complex'
    }
    
    # Subset for faster experiments during development
    for name, info in datasets_info.items():
        # Use 10% of training data for development, full for final runs
        subset_size = len(info['train']) // 10
        info['train_subset'] = Subset(info['train'], range(subset_size))
    
    return datasets_info

def prepare_nlp_datasets():
    """Prepare NLP datasets"""
    datasets_info = {}
    
    # IMDB sentiment analysis
    try:
        imdb = load_dataset('imdb')
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
        
        imdb_tokenized = imdb.map(tokenize_function, batched=True)
        
        datasets_info['IMDB'] = {
            'train': imdb_tokenized['train'], 'test': imdb_tokenized['test'],
            'model_class': lambda: TransformerClassifier('distilbert-base-uncased', 2),
            'num_classes': 2, 'complexity': 'medium', 'tokenizer': tokenizer
        }
    except Exception as e:
        logger.warning(f"Failed to load IMDB dataset: {e}")
    
    # AG News classification
    try:
        ag_news = load_dataset('ag_news')
        ag_news_tokenized = ag_news.map(tokenize_function, batched=True)
        
        datasets_info['AG_News'] = {
            'train': ag_news_tokenized['train'], 'test': ag_news_tokenized['test'],
            'model_class': lambda: TransformerClassifier('distilbert-base-uncased', 4),
            'num_classes': 4, 'complexity': 'simple', 'tokenizer': tokenizer
        }
    except Exception as e:
        logger.warning(f"Failed to load AG News dataset: {e}")
    
    return datasets_info

# =============================================================================
# SYSTEM MONITORING
# =============================================================================

class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.measurements = []
        self.start_time = None
    
    def start_monitoring(self):
        self.start_time = time.time()
        self.initial_memory = self.get_memory_usage()
    
    def record_measurement(self):
        measurement = {
            'timestamp': time.time() - self.start_time if self.start_time else 0,
            'memory_usage': self.get_memory_usage(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        # GPU monitoring if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                measurement.update({
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_utilization': gpu.load * 100
                })
        except ImportError:
            pass
        
        self.measurements.append(measurement)
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_summary(self):
        if not self.measurements:
            return {}
        
        df = pd.DataFrame(self.measurements)
        return {
            'peak_memory_mb': df['memory_usage'].max(),
            'avg_memory_mb': df['memory_usage'].mean(),
            'peak_cpu_percent': df['cpu_percent'].max(),
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'duration_seconds': df['timestamp'].max() if len(df) > 0 else 0
        }

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_vision_model(model, train_loader, val_loader, optimizer, criterion, 
                      max_epochs, device, monitor, tracker):
    """Train vision model with detailed monitoring"""
    
    model.train()
    best_val_acc = 0
    patience_counter = 0
    epoch_results = []
    
    monitor.start_monitoring()
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Record system metrics every 10 batches
            if batch_idx % 10 == 0:
                monitor.record_measurement()
        
        # Validation
        val_acc, val_loss = evaluate_vision_model(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        train_acc = train_correct / train_total
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_time': epoch_time
        }
        epoch_results.append(epoch_result)
        
        logger.info(f"Epoch {epoch+1}/{max_epochs}: "
                   f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return epoch_results, best_val_acc

def evaluate_vision_model(model, data_loader, criterion, device):
    """Evaluate vision model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total, total_loss / len(data_loader)

def train_nlp_model(model, train_dataset, val_dataset, optimizer, max_epochs, 
                   device, monitor, tracker, tokenizer):
    """Train NLP model using Hugging Face trainer"""
    
    training_args = TrainingArguments(
        output_dir='./tmp_trainer',
        num_train_epochs=max_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to=None  # Disable wandb/tensorboard
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": np.mean(predictions == labels)}
    
    # Custom data collator for our transformer model
    def data_collator(features):
        batch = {}
        batch['input_ids'] = torch.tensor([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features])
        batch['labels'] = torch.tensor([f['label'] for f in features])
        return batch
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    monitor.start_monitoring()
    
    # Train model
    trainer.train()
    
    # Get final evaluation
    eval_results = trainer.evaluate()
    
    return trainer.state.log_history, eval_results['eval_accuracy']

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_single_experiment(dataset_name, dataset_info, optimizer_name, optimizer_fn, 
                         seed, experiment_id):
    """Run a single experiment configuration"""
    
    logger.info(f"Starting experiment: {experiment_id}")
    
    # Set random seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Initialize monitoring
    monitor = SystemMonitor()
    tracker = EmissionsTracker(
        project_name=experiment_id,
        output_dir=str(CONFIG['output_dir'] / 'emissions'),
        log_level='error'  # Reduce logging noise
    )
    
    try:
        # Start emissions tracking
        tracker.start()
        
        # Create model and optimizer
        model = dataset_info['model_class']().to(CONFIG['device'])
        optimizer = optimizer_fn(model.parameters())
        
        # Prepare data
        if 'tokenizer' in dataset_info:  # NLP dataset
            # For NLP, use smaller subsets to manage training time
            train_size = min(5000, len(dataset_info['train']))
            val_size = min(1000, len(dataset_info['test']))
            
            train_subset = dataset_info['train'].select(range(train_size))
            val_subset = dataset_info['test'].select(range(val_size))
            
            # Train NLP model
            epoch_results, final_accuracy = train_nlp_model(
                model, train_subset, val_subset, optimizer,
                CONFIG['max_epochs']['nlp'], CONFIG['device'],
                monitor, tracker, dataset_info['tokenizer']
            )
            
        else:  # Vision dataset
            # Create data loaders
            batch_size = CONFIG['batch_sizes']['medium']
            
            # Use subset for development, full dataset for final runs
            train_dataset = dataset_info['train_subset']  # Change to 'train' for full runs
            
            # Split into train/val
            val_size = int(CONFIG['validation_split'] * len(train_dataset))
            train_size = len(train_dataset) - val_size
            
            train_data, val_data = torch.utils.data.random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            
            # Train vision model
            criterion = nn.CrossEntropyLoss()
            epoch_results, final_accuracy = train_vision_model(
                model, train_loader, val_loader, optimizer, criterion,
                CONFIG['max_epochs']['vision'], CONFIG['device'],
                monitor, tracker
            )
        
        # Stop tracking and get results
        emissions_data = tracker.stop()
        system_summary = monitor.get_summary()
        
        # Compile results
        result = {
            'experiment_id': experiment_id,
            'dataset': dataset_name,
            'optimizer': optimizer_name,
            'seed': seed,
            'final_accuracy': final_accuracy,
            'num_epochs': len(epoch_results),
            'total_duration': system_summary.get('duration_seconds', 0),
            'peak_memory_mb': system_summary.get('peak_memory_mb', 0),
            'avg_memory_mb': system_summary.get('avg_memory_mb', 0),
            'emissions_kg': emissions_data if emissions_data else 0,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'dataset_complexity': dataset_info['complexity'],
            'convergence_epoch': next((i for i, r in enumerate(epoch_results) 
                                     if r['val_acc'] >= 0.95 * final_accuracy), 
                                    len(epoch_results)),
            'epoch_details': epoch_results
        }
        
        logger.info(f"Completed experiment: {experiment_id} - "
                   f"Final Acc: {final_accuracy:.4f}, "
                   f"Duration: {result['total_duration']:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        tracker.stop()
        return None
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_comprehensive_study():
    """Run the complete comprehensive study"""
    
    # Create output directories
    CONFIG['output_dir'].mkdir(exist_ok=True)
    (CONFIG['output_dir'] / 'emissions').mkdir(exist_ok=True)
    CONFIG['cache_dir'].mkdir(exist_ok=True)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    vision_datasets = prepare_vision_datasets()
    nlp_datasets = prepare_nlp_datasets()
    all_datasets = {**vision_datasets, **nlp_datasets}
    
    logger.info(f"Prepared {len(all_datasets)} datasets: {list(all_datasets.keys())}")
    logger.info(f"Will test {len(OPTIMIZERS)} optimizers: {list(OPTIMIZERS.keys())}")
    logger.info(f"Running {len(CONFIG['seeds'])} seeds per configuration")
    
    total_experiments = len(all_datasets) * len(OPTIMIZERS) * len(CONFIG['seeds'])
    logger.info(f"Total experiments to run: {total_experiments}")
    
    # Run all experiments
    results = []
    experiment_count = 0
    
    for dataset_name, dataset_info in all_datasets.items():
        for optimizer_name, optimizer_fn in OPTIMIZERS.items():
            for seed in CONFIG['seeds']:
                experiment_count += 1
                experiment_id = f"{dataset_name}_{optimizer_name}_seed{seed:02d}"
                
                logger.info(f"Experiment {experiment_count}/{total_experiments}: {experiment_id}")
                
                result = run_single_experiment(
                    dataset_name, dataset_info, optimizer_name, 
                    optimizer_fn, seed, experiment_id
                )
                
                if result:
                    results.append(result)
                    
                    # Save intermediate results
                    if len(results) % 10 == 0:
                        df = pd.DataFrame(results)
                        df.to_csv(CONFIG['output_dir'] / 'intermediate_results.csv', index=False)
                        logger.info(f"Saved intermediate results ({len(results)} experiments)")
    
    # Save final results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(CONFIG['output_dir'] / 'comprehensive_results.csv', index=False)
        
        # Save detailed epoch results separately
        epoch_details = []
        for result in results:
            for epoch_data in result['epoch_details']:
                epoch_record = {
                    'experiment_id': result['experiment_id'],
                    'dataset': result['dataset'],
                    'optimizer': result['optimizer'],
                    'seed': result['seed'],
                    **epoch_data
                }
                epoch_details.append(epoch_record)
        
        epoch_df = pd.DataFrame(epoch_details)
        epoch_df.to_csv(CONFIG['output_dir'] / 'epoch_details.csv', index=False)
        
        # Save experiment configuration
        config_dict = {
            'total_experiments': len(results),
            'datasets': list(all_datasets.keys()),
            'optimizers': list(OPTIMIZERS.keys()),
            'seeds': CONFIG['seeds'],
            'device': str(CONFIG['device']),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(CONFIG['output_dir'] / 'experiment_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"✅ Study complete! {len(results)} successful experiments")
        logger.info(f"Results saved to {CONFIG['output_dir']}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        summary_stats = df.groupby(['dataset', 'optimizer']).agg({
            'final_accuracy': ['mean', 'std'],
            'total_duration': ['mean', 'std'],
            'emissions_kg': ['mean', 'std'],
            'peak_memory_mb': ['mean', 'std']
        }).round(4)
        
        print(summary_stats)
        
    else:
        logger.error("No successful experiments completed!")

if __name__ == "__main__":
    logger.info("Starting Comprehensive Neural Network Optimizer Energy Study")
    logger.info(f"Using device: {CONFIG['device']}")
    
    # Check system resources
    logger.info(f"Available CPU cores: {psutil.cpu_count()}")
    logger.info(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    run_comprehensive_study()