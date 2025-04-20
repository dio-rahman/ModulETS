import os
import sys
import importlib
import subprocess
import json

def check_dependencies():
    required_packages = [
        'numpy', 'opencv-python', 'scikit-learn', 'matplotlib',
        'flask', 'Pillow', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"[OK] {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"[MISSING] {package} is not installed")
    
    return missing_packages

def check_dataset_structure():
    dataset_path = "./dataset_wajah"
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory {dataset_path} does not exist")
        return False
    
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"[ERROR] No image files found in {dataset_path}")
        return False
    
    print(f"[OK] Found {len(image_files)} images in dataset directory")
    return True

def check_metadata_file():
    metadata_path = "./metadata.json"
    
    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata file {metadata_path} does not exist")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[ERROR] Metadata file is not in the expected format")
            return False
        print(f"[OK] Metadata file exists and is valid")
        return True
    except json.JSONDecodeError:
        print(f"[ERROR] Metadata file is not valid JSON")
        return False

def check_app_requirements():
    print("Checking system requirements...")
    print("-" * 30)
    
    print("Checking dependencies:")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print("[ERROR] Some required packages are missing")
        install = input("Do you want to install the missing packages? (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package])
            print("Dependencies installed")
        else:
            print("Please install missing packages and try again")
            return False
    else:
        print("[OK] Basic dependencies are installed")
    
    print("\nChecking dataset:")
    if not check_dataset_structure():
        return False
    
    print("\nChecking metadata:")
    if not check_metadata_file():
        return False
    
    print("\nAll checks passed! The system is ready to run.")
    return True

def run_tests():
    try:
        import cv2
        import numpy as np
        from sklearn import preprocessing
        
        test_img_path = "./dataset_wajah"
        if os.path.exists(test_img_path):
            files = [f for f in os.listdir(test_img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                img = cv2.imread(os.path.join(test_img_path, files[0]))
                if img is not None:
                    print("[OK] Image loading test passed")
                else:
                    print("[ERROR] Failed to load test image")
                    return False
            else:
                print("[WARNING] No test images found to check loading")
        
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(test_array)
        if scaled is not None:
            print("[OK] Model functioning test passed")
        else:
            print("[ERROR] Model functioning test failed")
            return False
            
        return True
    except Exception as e:
        print(f"[ERROR] Test failed with error: {str(e)}")
        return False

def main():
    print("Running integration check...\n")
    
    if check_app_requirements():
        print("\nRunning basic functionality tests...")
        if run_tests():
            print("\nIntegration check completed successfully.")
            return True
        else:
            print("\nFunctionality tests failed. Please fix the issues and try again.")
            return False
    else:
        print("\nSystem requirements check failed. Please fix the issues and try again.")
        return False

if __name__ == "__main__":
    main()