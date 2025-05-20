import os
import sys
import subprocess
import pkg_resources

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

# Function to install requirements
def install_requirements():
    requirements_file = os.path.join(script_dir, "requirements.txt")
    
    if not os.path.exists(requirements_file):
        print(f"Requirements file not found at {requirements_file}")
        return False
    
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

# Main installation function
def main():
    print("Starting DIA TTS ComfyUI node installation...")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install them manually.")
        return
    
    print("\nDIA TTS ComfyUI node installation completed!")
    print("\nNotes:")
    print("- The first time you use the node, it will download the DIA model (this may take some time)")
    print("- Make sure your ComfyUI has access to sufficient GPU memory (10+ GB recommended)")
    print("- For best results, provide a 5-10 second audio sample with matching transcript")

if __name__ == "__main__":
    main()
