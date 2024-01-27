import torch
import matplotlib.pyplot as plt

def create_optimizer_with_exponential_lrs(model, min_lr, max_lr, num_layers, opti_const, opti_kwag):
    # Calculate the multiplicative factor
    factor = (max_lr / min_lr) ** (1 / (num_layers - 1))
    
    # Generate learning rates for each layer
    layer_lrs = [min_lr * (factor ** i) for i in range(num_layers)]
    
    # Create parameter groups
    optimizer_param_groups = []
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        optimizer_param_groups.append({'params': layer.parameters(), 'lr': layer_lrs[i]})
    
    # Add the fully connected layer with the maximum learning rate
    optimizer_param_groups.append({'params': model.fc.parameters(), 'lr': max_lr})
    
    # Optionally, include other layers with the minimum learning rate
    base_params = list(model.conv1.parameters()) + list(model.bn1.parameters()) + \
                  list(model.relu.parameters()) + list(model.maxpool.parameters())
    optimizer_param_groups.append({'params': base_params, 'lr': min_lr})
    
    # Create the optimizer
    if opti_kwag is not None:
        optimizer = opti_const(optimizer_param_groups, **opti_kwag)
    else:
        optimizer = opti_const(optimizer_param_groups)

    return optimizer

def display_image_grid(dataset, dataloader, model, device, grid_size=5):
    """
    Display a grid of images with predictions.
    """

    model.eval()  # Set the model to evaluation mode
    classes = dataset.classes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    image_count = 0

    with torch.no_grad():
        for image, label in dataloader:
            output = model(image.to(device))
            predictions = output.max(-1)[1].cpu()

            for i in range(len(image)):
                if image_count == grid_size ** 2:
                    break
                ax = axes[image_count]
                img = image[i].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1 range

                ax.imshow(img)
                ax.axis('off')
                ax.set_title(classes[predictions[i]])

                image_count += 1

            if image_count == grid_size ** 2:
                break

    plt.tight_layout()
    plt.show()
