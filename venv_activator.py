import os
import sys
import site

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, "venv")

def activate_venv():
    """Activate the virtual environment if not already activated"""
    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return
    
    # Determine the site-packages directory
    if os.name == 'nt':  # Windows
        site_packages = os.path.join(VENV_DIR, 'Lib', 'site-packages')
        bin_dir = os.path.join(VENV_DIR, 'Scripts')
    else:  # Unix
        lib_path = [d for d in os.listdir(VENV_DIR) if d.startswith('lib')][0]
        python_path = [d for d in os.listdir(os.path.join(VENV_DIR, lib_path)) if d.startswith('python')][0]
        site_packages = os.path.join(VENV_DIR, lib_path, python_path, 'site-packages')
        bin_dir = os.path.join(VENV_DIR, 'bin')
    
    # Add the site-packages directory to sys.path
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    
    # Add the bin directory to PATH
    if bin_dir not in os.environ['PATH']:
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']
    
    # Update sys.prefix and sys.exec_prefix
    sys.prefix = VENV_DIR
    if hasattr(sys, 'base_prefix'):
        sys.base_prefix = sys.prefix
    if hasattr(sys, 'real_prefix'):
        sys.real_prefix = sys.prefix
    
    # Update site.py
    site.addsitedir(site_packages)
