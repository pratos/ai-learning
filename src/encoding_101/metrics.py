import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


def calculate_mar_at_k(dataloader, model, k=5, device='cuda'):
    """
    Calculate MAR@k for a dataset using a trained model
    
    Args:
        dataloader: DataLoader for the dataset
        model: Trained model with encode method
        k: Number of nearest neighbors to consider
        device: Device to use for computation
        
    Returns:
        mar_at_k: Mean Average Recall@k
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            # Get embeddings
            z = model.encode(x)
            embeddings.append(z.cpu())
            labels.append(y.cpu())
    
    # Concatenate all embeddings and labels
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Calculate MAR@k
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    cos_sim = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    # Set diagonal to -inf to exclude self-comparisons
    cos_sim.fill_diagonal_(-float('inf'))
    
    # Get top-k indices for each embedding
    _, topk_indices = cos_sim.topk(k=k, dim=1)
    
    # Compute recall@k for each query
    recalls = []
    
    # Convert to numpy for easier handling
    labels_np = labels.numpy()
    topk_indices_np = topk_indices.numpy()
    
    for i, query_label in enumerate(labels_np):
        # Get labels of the top-k nearest neighbors
        neighbor_labels = labels_np[topk_indices_np[i]]
        
        # Count how many are from the same class
        relevant_retrieved = (neighbor_labels == query_label).sum()
        
        # Count total number of relevant items in the dataset (excluding self)
        total_relevant = (labels_np == query_label).sum() - 1
        
        # Calculate recall for this query
        if total_relevant > 0:
            recall = min(relevant_retrieved / total_relevant, 1.0)
            recalls.append(recall)
    
    # Calculate mean recall
    if recalls:
        return sum(recalls) / len(recalls)
    else:
        return 0.0


def visualize_mar_at_k(dataloader, model, output_dir=None, samples_per_class=10, k=5, device='cuda', 
                       class_names=None):
    """
    Visualize MAR@k by showing query images and their nearest neighbors
    
    Args:
        dataloader: DataLoader for the dataset
        model: Trained model with encode method
        output_dir: Directory to save visualization images
        samples_per_class: Number of samples to visualize per class
        k: Number of nearest neighbors to consider
        device: Device to use for computation
        class_names: List of class names (optional)
    
    Returns:
        mar_at_k: Mean Average Recall@k
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # First pass: collect all embeddings and data
    all_images = []
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            # Get embeddings
            z = model.encode(x)
            all_embeddings.append(z.cpu())
            all_labels.append(y.cpu())
            all_images.append(x.cpu())
    
    # Concatenate
    all_images = torch.cat(all_images, dim=0)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute similarity 
    normalized_embeddings = F.normalize(all_embeddings, p=2, dim=1)
    cos_sim = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    # Set diagonal to -inf to exclude self-comparisons
    cos_sim.fill_diagonal_(-float('inf'))
    
    # Get indices of samples for each class
    class_indices = {}
    for i in range(10):  # Assuming 10 classes in CIFAR-10
        class_indices[i] = (all_labels == i).nonzero(as_tuple=True)[0]
        
    # Select random samples for each class
    selected_indices = []
    for class_idx in range(10):
        indices = class_indices[class_idx]
        if len(indices) > samples_per_class:
            # Randomly select samples_per_class samples
            selected = indices[torch.randperm(len(indices))[:samples_per_class]]
            selected_indices.append(selected)
        else:
            # Use all available samples
            selected_indices.append(indices)
    
    # Flatten the list of indices
    selected_indices = torch.cat(selected_indices)
    
    # Calculate MAR@k
    total_recalls = []
    
    # Create a figure for each query image
    for i, query_idx in enumerate(selected_indices):
        query_img = all_images[query_idx]
        query_label = all_labels[query_idx].item()
        
        # Get top-k nearest neighbors
        similarities = cos_sim[query_idx]
        _, nn_indices = similarities.topk(k)
        nn_labels = all_labels[nn_indices]
        nn_images = all_images[nn_indices]
        
        # Calculate recall for this query
        total_relevant = (all_labels == query_label).sum().item() - 1  # Excluding self
        relevant_retrieved = (nn_labels == query_label).sum().item()
        
        recall = min(relevant_retrieved / total_relevant, 1.0) if total_relevant > 0 else 0
        total_recalls.append(recall)
        
        # Create a visualization showing the query and its neighbors
        fig, axes = plt.subplots(1, k+1, figsize=(15, 3))
        
        # Query image
        query_img_np = query_img.permute(1, 2, 0).numpy()
        query_img_np = np.clip(query_img_np, 0, 1)
        axes[0].imshow(query_img_np)
        axes[0].set_title(f"Query\n{class_names[query_label]}")
        axes[0].axis('off')
        
        # Neighbor images
        for j in range(k):
            nn_img = nn_images[j]
            nn_label = nn_labels[j].item()
            
            nn_img_np = nn_img.permute(1, 2, 0).numpy()
            nn_img_np = np.clip(nn_img_np, 0, 1)
            
            # Add red/green border based on whether it's the same class
            is_same_class = nn_label == query_label
            border_color = 'green' if is_same_class else 'red'
            
            # Display with colored border
            axes[j+1].imshow(nn_img_np)
            axes[j+1].spines['top'].set_color(border_color)
            axes[j+1].spines['bottom'].set_color(border_color)
            axes[j+1].spines['left'].set_color(border_color)
            axes[j+1].spines['right'].set_color(border_color)
            axes[j+1].spines['top'].set_linewidth(5)
            axes[j+1].spines['bottom'].set_linewidth(5)
            axes[j+1].spines['left'].set_linewidth(5)
            axes[j+1].spines['right'].set_linewidth(5)
            
            axes[j+1].set_title(f"NN {j+1}\n{class_names[nn_label]}")
            axes[j+1].axis('off')
        
        plt.suptitle(f"Query: {class_names[query_label]}, Recall@{k}: {recall:.2f}", fontsize=16)
        plt.tight_layout()
        
        # Save or show
        if output_dir is not None:
            class_dir = os.path.join(output_dir, f"class_{query_label}")
            os.makedirs(class_dir, exist_ok=True)
            plt.savefig(os.path.join(class_dir, f"sample_{i % samples_per_class}.png"), dpi=150)
            plt.close(fig)
        else:
            plt.show()
    
    # Calculate mean recall
    mar_at_k = sum(total_recalls) / len(total_recalls) if total_recalls else 0
    
    # Final summary visualization
    if output_dir is not None:
        # Create a summary bar chart for each class
        class_recalls = {}
        for i, idx in enumerate(selected_indices):
            class_label = all_labels[idx].item()
            if class_label not in class_recalls:
                class_recalls[class_label] = []
            class_recalls[class_label].append(total_recalls[i])
        
        # Plot mean recall per class
        plt.figure(figsize=(12, 6))
        classes = list(range(10))
        class_mean_recalls = [np.mean(class_recalls[c]) for c in classes]
        
        bars = plt.bar(classes, class_mean_recalls)
        plt.title(f'MAR@{k} by Class (Overall: {mar_at_k:.4f})', fontsize=16)
        plt.xlabel('Class')
        plt.ylabel(f'Mean Recall@{k}')
        plt.xticks(classes, [class_names[i] for i in classes], rotation=45)
        
        # Color bars by recall value
        for i, bar in enumerate(bars):
            if class_mean_recalls[i] > 0.8:
                bar.set_color('darkgreen')
            elif class_mean_recalls[i] > 0.6:
                bar.set_color('green')
            elif class_mean_recalls[i] > 0.4:
                bar.set_color('orange')
            elif class_mean_recalls[i] > 0.2:
                bar.set_color('darkorange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'mar_at_{k}_summary.png'), dpi=150)
        plt.close()
        
        logger.info(f"Saved MAR@{k} visualizations to {output_dir}")
        logger.info(f"Overall MAR@{k}: {mar_at_k:.4f}")
        for c in classes:
            logger.info(f"Class {class_names[c]}: {np.mean(class_recalls[c]):.4f}")
    
    return mar_at_k 