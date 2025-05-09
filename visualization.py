import numpy as np
import matplotlib.pyplot as plt
import cv2

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """
    Overlay a heatmap on an image for visualization.
    
    Args:
        img_array: The original image as a numpy array
        heatmap: The heatmap to overlay
        alpha: Transparency factor
        
    Returns:
        The original image with heatmap overlay
    """
    # Convert heatmap to RGB colormap (jet)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(np.uint8(img_array), cv2.COLOR_RGB2BGR)
    else:
        # Handle grayscale images
        img_bgr = cv2.cvtColor(np.uint8(img_array), cv2.COLOR_GRAY2BGR)
    
    # Overlay the heatmap on original image
    superimposed_img = cv2.addWeighted(
        img_bgr,
        1-alpha,  # Original image weight
        heatmap,
        alpha,    # Heatmap weight
        0         # Scalar added to each sum
    )
    
    # Convert back to RGB for matplotlib
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

def plot_training_history(history, save_path=None):
    """
    Plot training history metrics.
    
    Args:
        history: Keras history object from model.fit()
        save_path: Path to save the plot image
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()