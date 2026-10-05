for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    
    out = model(x_train_tensor, adj_train_tensor)
    mse_loss = criterion(out, x_train_tensor)
    smoothness_loss = torch.mean(torch.diff(out, dim=0) ** 2)
    total_loss = mse_loss + 0.5 * smoothness_loss
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} (MSE: {mse_loss.item():.6f})")