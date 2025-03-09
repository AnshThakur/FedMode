from Model_IHM import *
import torch
import torch.optim as optim
import numpy as np

# Define the quadratic Bezier curve
def bezier_curve(theta_i, theta_j, theta_m, alpha):
    return [(1 - alpha)**2 * ti + 2 * alpha * (1 - alpha) * tm + alpha**2 * tj for ti, tm, tj in zip(theta_i, theta_m, theta_j)]







def de_casteljau(control_points, t):
    points = control_points
    while len(points) > 1:
        new_points = [(1 - t) * points[i] + t * points[i + 1] for i in range(len(points) - 1)]
        points = new_points
    return points[0]


def compute_loss(model, criterion, data_loader, params, device):
    """Computes loss using a given set of parameters without breaking the computation graph."""
    with torch.no_grad():  # Only disable gradients for loss evaluation
        for param, new_param in zip(model.parameters(), params):
            param.copy_(new_param)  # Update parameters without breaking graph
        
        model.eval()
        total_loss = 0.0
        total_batches = 0
        for data, target in data_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)[:, 0]
            loss = criterion(output, target)
            total_loss += loss.item()
            total_batches += 1
        
    return total_loss / total_batches

def learn_low_loss_bezier_curve(global_model, client_model, criterion, data_loader, num_samples=10, num_steps=2, lr=0.001):
    device = next(global_model.parameters()).device
    global_params = [param.clone().detach().to(device) for param in global_model.parameters()]
    local_params = [param.clone().detach().to(device) for param in client_model.parameters()]

    # Initialize control points as trainable parameters
    control_points = [torch.nn.Parameter(0.5 * (gp + lp)) for gp, lp in zip(global_params, local_params)]
    optimizer = optim.Adam(control_points, lr=lr)

    Alphas = np.linspace(0, 1, num_samples)

    for step in range(num_steps):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            t = np.random.choice(Alphas)
            
            # Compute sampled parameters without breaking the graph
            sampled_params = [de_casteljau([gp, cp, lp], t) for gp, cp, lp in zip(global_params, control_points, local_params)]
            
            # Update model parameters in-place
            for param, new_param in zip(global_model.parameters(), sampled_params):
                param.data.copy_(new_param)  # Avoid breaking the graph

            data, target = data.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = global_model(data)[:, 0]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Sample points along the optimized Bezier curve and find the one with the lowest loss
    best_loss = float('inf')
    best_params = None
    losses = []
    t_values = np.linspace(0, 1, num_samples)
    sampled_paths = []
    
    for t in t_values:
        sampled_params = [de_casteljau([gp, cp, lp], t) for gp, cp, lp in zip(global_params, control_points, local_params)]
        
        # Compute loss while keeping parameters intact
        loss = compute_loss(global_model, criterion, data_loader, sampled_params, device)
        sampled_paths.append(sampled_params)
        losses.append(loss)
        
        if loss < best_loss:
            best_loss = loss
            best_params = sampled_params

    return best_params, t_values, losses, sampled_paths



def client_update(global_model_params, train_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = LSTMClassifier(44,64,device).to(device)
    for param, global_param in zip(global_model.parameters(), global_model_params):
        param.data.copy_(global_param)
    
    client_model = LSTMClassifier(44,64,device).to(device)
    for param, global_param in zip(client_model.parameters(), global_model_params):
        param.data.copy_(global_param)

    optimizer = optim.Adam(client_model.parameters(), lr=0.001)
    # client_model,TL = train_model_with_proximity_reg(client_model, global_model,train_loader[0], criterion, optimizer,device=device)
    client_model,TL = train_model(client_model, train_loader[0], criterion, optimizer,device=device)

    best_params, t_values, losses, sampled_paths = learn_low_loss_bezier_curve(global_model, client_model, criterion, train_loader[0])
    return sampled_paths, losses,TL,client_model



def train_model(model, data_loader, criterion, optimizer,device, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        L=0
        for data, target in data_loader:
            data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)
            optimizer.zero_grad()
            output = model(data)[:,0]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            L=L+loss.item()
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {L/len(data_loader):.4f}')
    return model,L/len(data_loader)


def train_model_with_proximity_reg(model, global_model, data_loader, criterion, optimizer, mu=0.01, epochs=1, device='cuda'):
    model.train()

    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in data_loader:
            data, target = data.to(device).to(torch.float32), target.to(device).to(torch.float32) 
            optimizer.zero_grad()
            output = model(data)[:, 0] 
            loss = criterion(output, target)
            
            # Add proximal term
            prox_term = 0.0
            for param, global_param in zip(model.parameters(), global_model.parameters()):
                prox_term += (mu/2) * torch.norm(param - global_param) ** 2
            loss += prox_term
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        total_loss += avg_epoch_loss
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}')

    avg_total_loss = total_loss / epochs
    
    # Return updated model parameters and the average loss over all epochs
    return model, avg_total_loss