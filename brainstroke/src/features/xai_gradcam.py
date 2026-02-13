import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 

def get_img_array(img_path, size):
    """Utility function to load and preprocess an image from a path."""
    # Read the image using OpenCV (loads as BGR)
    img = cv2.imread(img_path) 
    
    # Convert from BGR to RGB immediately for consistent processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to the standard input size
    img = cv2.resize(img, size)
    
    # Normalize pixel values
    array = img.astype('float32') / 255.0 
    array = np.expand_dims(array, axis=0) 
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates the Grad-CAM heatmap based on the predicted class.
    (Code logic remains the same as it was already correct)
    """
    default_heatmap_size = img_array.shape[1:3] if img_array.ndim == 4 else (224, 224) 
    
    try:
        if not any(last_conv_layer_name in layer.name for layer in model.layers):
            return np.zeros(default_heatmap_size)

        grad_model = keras.models.Model(
            [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
                
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap_max = tf.math.reduce_max(heatmap)
        if heatmap_max == 0:
            heatmap = heatmap * 0
        else:
            heatmap = tf.maximum(heatmap, 0) / heatmap_max
            
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        return np.zeros(default_heatmap_size)


def save_and_display_gradcam(img_path, heatmap, alpha=0.5):
    """Creates the final superimposed image."""
    # Load original image as RGB for blending (Note: cv2.imread loads BGR)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # Rescale and apply colormap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

    # Superimpose the images
    superimposed_img = cv2.addWeighted(img, 1.0 - alpha, jet_heatmap, alpha, 0)
    
    # Streamlit expects RGB, and this is now an RGB numpy array
    return superimposed_img