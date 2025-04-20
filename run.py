import os
import sys
import subprocess
import time

def run_command(command, description):
    print(f"\n=== {description} ===")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Success: {description}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Details: {e.stderr}")
        return False, e.stderr

def main():
    DATASET_PATH = "./dataset_wajah"

    if not os.path.exists(DATASET_PATH):
        print(f"Creating dataset directory at {DATASET_PATH}...")
        os.makedirs(DATASET_PATH, exist_ok=True)
    
    image_files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in dataset directory {DATASET_PATH}. Please add images before continuing.")
        return
    
    success, _ = run_command("python create_metadata.py", "Generating metadata file")
    if not success:
        print("Failed to generate metadata. Aborting.")
        return
    
    success, _ = run_command("python integrate.py", "Running integration check")
    if not success:
        print("Integration check failed. Would you like to proceed anyway? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            print("Aborting.")
            return
    
    print("\n=== Starting the application ===")
    print("The application will now start. Press Ctrl+C to stop.")
    try:
        subprocess.run("python app.py", shell=True)
    except KeyboardInterrupt:
        print("\nApplication stopped.")

if __name__ == "__main__":
    main()