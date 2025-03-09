import torch.nn.functional as F
import torch 


def find_common_intersection(client_paths, client_losses, prev_global_params, num_steps=300, lr=0.001, lambda_reg=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_clients = len(client_paths)
    del1=int(len(client_paths[0])*0.25)
    n_samples = len(client_paths[0])-del1
    n_params = len(client_paths[0][0])

    # Initialize intersection point as the average of initial sampled points
    intersection_point = [(torch.mean(torch.stack([client_paths[client_idx][t_idx+del1][param_idx] for client_idx in range(n_clients)]), dim=0)
     + torch.randn_like(prev_global_params[param_idx]) * 0.025)  # Small perturbation
    .detach().clone().to(device).requires_grad_(True)
    for t_idx in range(n_samples)
    for param_idx in range(n_params)]

    optimizer = torch.optim.AdamW(intersection_point, lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()  # Zero the gradients at the start of each step
        
        total_loss = 0
        
        # Shuffle the samples to improve convergence
        indices = torch.randperm(n_samples * n_clients)

        for index in indices:
            client_idx = index // n_samples
            t_idx = index % n_samples
            sampled_params = client_paths[client_idx][t_idx+del1]
            loss = client_losses[client_idx][t_idx+del1]
            
            # Compute the loss for this sample
            sample_loss = 1/(loss+0.0000001) * compute_weight(intersection_point, sampled_params)
            total_loss += sample_loss
        
        # Backward pass for the primary loss
        

        reg_loss = - sum(F.mse_loss(intersection_point[param_idx], prev_global_params[param_idx]) for param_idx in range(n_params)) # For smoother transitions, we can make regularisation positive. This might help in extreme cases
        total_loss=total_loss+lambda_reg*reg_loss        
        total_loss.backward(retain_graph=True)


        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}/{num_steps}, Loss: {total_loss.item():.6f} Reg Loss: {reg_loss.item():.6f}")

    return intersection_point

def compute_weight(intersection_point, sampled_params):
    return sum(F.mse_loss(ip, sp) for ip, sp in zip(intersection_point, sampled_params))



def aggregate_models(global_model, intersection_point):
    new_state_dict = {k: v.clone() for k, v in zip(global_model.state_dict().keys(), intersection_point)}
    global_model.load_state_dict(new_state_dict)
    return global_model
