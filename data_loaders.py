

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split



def create_non_iid_loaders(data_path, labels_path, num_clients, batch_size=64, min_samples_per_class=10, alpha=0.2, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # Load data and labels, expects numpy arrays
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    # Ensure data and labels are in the correct shape
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples"
    
    # Initialize lists to hold data loaders for each client
    client_loaders = []
    
    # Shuffle data and labels together
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    # Separate indices by class
    label_indices = [np.where(labels == i)[0] for i in range(2)]
    client_data_indices = [[] for _ in range(num_clients)]
    
    # Assign minimum samples per class to each client
    for i, label_idx in enumerate(label_indices):
        np.random.shuffle(label_idx)
        for client_idx in range(num_clients):
            allocated_samples = min_samples_per_class if len(label_idx) >= min_samples_per_class else len(label_idx)
            client_data_indices[client_idx].extend(label_idx[:allocated_samples].tolist())
            label_idx = label_idx[allocated_samples:]
    
    # Distribute remaining samples non-uniformly among clients using Dirichlet distribution
    for i, label_idx in enumerate(label_indices):
        if len(label_idx) == 0:
            continue
        # Use the alpha parameter to control the degree of non-IID
        portions = np.random.dirichlet(np.ones(num_clients) * alpha, size=1)[0]
        split_indices = np.split(label_idx, (np.cumsum(portions)[:-1] * len(label_idx)).astype(int))
        for client_idx, client_split in enumerate(split_indices):
            client_data_indices[client_idx].extend(client_split.tolist())
    
    LOADERS=[]
    # Create data loaders for each client
    for client_idx in range(num_clients):
        client_indices = client_data_indices[client_idx]
        client_data = torch.tensor(data[client_indices], dtype=torch.float32)
        client_labels = torch.tensor(labels[client_indices], dtype=torch.long)
        
        print(client_idx)
        print(client_data.shape)
        # Separate positive and negative examples
        positive_indices = torch.where(client_labels == 1)[0]
        negative_indices = torch.where(client_labels == 0)[0]
        
        # Ensure that each split has positive examples
        def ensure_split(split_size):
            pos_split_size = max(1, int(split_size * len(positive_indices) / len(client_labels)))
            neg_split_size = split_size - pos_split_size
            pos_samples = positive_indices[:pos_split_size]
            neg_samples = negative_indices[:neg_split_size]
            return torch.cat((pos_samples, neg_samples)), pos_split_size
        
        # Split dataset into train, val, and test
        train_size = int(train_ratio * len(client_labels))
        val_size = int(val_ratio * len(client_labels))
        test_size = len(client_labels) - train_size - val_size
        
        train_indices, train_pos_count = ensure_split(train_size)
        val_indices, val_pos_count = ensure_split(val_size)
        test_indices, test_pos_count = ensure_split(test_size)
        print((train_pos_count+val_pos_count+test_pos_count)/len(client_data))

        print('------------------------------')        
        # Create datasets
        train_set = TensorDataset(client_data[train_indices], client_labels[train_indices])
        val_set = TensorDataset(client_data[val_indices], client_labels[val_indices])
        test_set = TensorDataset(client_data[test_indices], client_labels[test_indices])
        
        # Create loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        # Store the loaders in a dictionary for this client
        LOADERS.append([train_loader,val_loader,test_loader])
    
    return LOADERS