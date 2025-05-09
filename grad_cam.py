import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from visualization import overlay_heatmap
import os
import json

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block_16_project'):
    """
    Generate Grad-CAM heatmap for model visualization.
    Uses the correct layer name for MobileNetV2.
    """
    # First, ensure the layer exists in the model
    layer_names = [layer.name for layer in model.layers]
    if last_conv_layer_name not in layer_names:
        # Fallback to the last convolutional layer if specified one doesn't exist
        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
        if conv_layers:
            last_conv_layer_name = conv_layers[-1] 
            print(f"Using {last_conv_layer_name} as the last convolutional layer.")
        else:
            raise ValueError("Could not find a suitable convolutional layer for Grad-CAM")

    # Build a model that maps the input image to the activations of the last conv layer and output
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Mean intensity of the gradient over the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply channel-wise weights with the convolution outputs
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def analyze_and_visualize(model_path, img_path, output_dir="results"):
    """
    Analyze an image using the model and visualize the results with Grad-CAM.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    try:
        model = load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Load and preprocess image
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Successfully loaded image from {img_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

    # Generate heatmap with the correct layer name for MobileNetV2
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='block_16_project')

    # Resize and convert heatmap
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    
    # Overlay heatmap
    original_img = np.uint8(img_array[0] * 255)
    superimposed_img = overlay_heatmap(original_img, heatmap)

    # Save figures
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Highlighted Areas (Grad-CAM)')
    plt.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{base_filename}_analysis.png")
    plt.savefig(output_path)
    print(f"Analysis visualization saved to {output_path}")
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Try to load class indices
    class_indices = None
    try:
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
    except:
        pass
        
    # Determine result with correct class interpretation
    if prediction > 0.5:
        # Higher probability indicates the second class (typically index 1)
        result = "Compliant (Organized Store)" if not class_indices else f"Class 1 ({list(class_indices.keys())[1]})"
    else:
        # Lower probability indicates the first class (typically index 0)
        result = "Non-compliant (Cluttered Store)" if not class_indices else f"Class 0 ({list(class_indices.keys())[0]})"
    
    result_with_confidence = f"{result} - Confidence: {prediction:.2f}"
    print(result_with_confidence)
    
    return result_with_confidence