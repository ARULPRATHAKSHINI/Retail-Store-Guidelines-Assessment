# In predict.py
import os
import sys
from grad_cam import analyze_and_visualize

def main():
    # Check if model exists
    model_path = 'models/best_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please run train.py first to train and save the model.")
        sys.exit(1)
    
    # Check if image exists
    img_path =r'C:\Users\Arul Sree\OneDrive\Desktop\store\retail-env\test2.jpg'
    if not os.path.exists(img_path):
        print(f"Error: Image file {img_path} not found!")
        print("Please place a test image named 'test1.jpeg' in the current directory.")
        sys.exit(1)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Analyze image
    print(f"Analyzing image {img_path} with model {model_path}...")
    result = analyze_and_visualize(model_path, img_path, output_dir='results')
    print("\nAnalysis Result:")
    print(result)
    print("\nVisualization saved to results directory.")

if __name__ == "__main__":
    main()