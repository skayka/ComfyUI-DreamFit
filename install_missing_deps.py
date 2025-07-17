#!/usr/bin/env python3
"""
Install only missing dependencies from requirements.txt
without touching existing packages
"""

import subprocess
import sys
import pkg_resources

def get_installed_packages():
    """Get a dict of installed packages and their versions"""
    installed = {}
    for dist in pkg_resources.working_set:
        installed[dist.key] = dist.version
    return installed

def parse_requirement(req_line):
    """Parse a requirement line to get package name"""
    req_line = req_line.strip()
    if not req_line or req_line.startswith('#'):
        return None
    
    # Remove version specifiers
    for op in ['==', '>=', '<=', '>', '<', '~=', '!=']:
        if op in req_line:
            return req_line.split(op)[0].strip()
    return req_line.strip()

def main():
    installed = get_installed_packages()
    missing_packages = []
    
    # Read requirements
    with open('DreamFit-official/requirements.txt', 'r') as f:
        requirements = f.readlines()
    
    print("Checking installed packages...")
    print("-" * 50)
    
    for req in requirements:
        pkg_name = parse_requirement(req)
        if not pkg_name:
            continue
            
        # Normalize package name (e.g., opencv-python -> opencv_python)
        normalized_name = pkg_name.lower().replace('-', '_')
        
        if normalized_name not in installed and pkg_name.lower() not in installed:
            missing_packages.append(req.strip())
            print(f"❌ Missing: {pkg_name}")
        else:
            version = installed.get(normalized_name) or installed.get(pkg_name.lower())
            print(f"✅ Found: {pkg_name} (version: {version})")
    
    print("-" * 50)
    
    if not missing_packages:
        print("All dependencies are already installed!")
        return
    
    print(f"\nFound {len(missing_packages)} missing packages:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    
    response = input("\nDo you want to install these missing packages? (y/n): ")
    if response.lower() == 'y':
        print("\nInstalling missing packages...")
        for pkg in missing_packages:
            print(f"\nInstalling {pkg}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', pkg])
                print(f"✅ Successfully installed {pkg}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {pkg}")
                # Try without version constraint
                pkg_name = parse_requirement(pkg)
                if pkg_name:
                    print(f"Trying to install {pkg_name} without version constraint...")
                    try:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', pkg_name])
                        print(f"✅ Successfully installed {pkg_name}")
                    except subprocess.CalledProcessError:
                        print(f"❌ Failed to install {pkg_name}")
    else:
        print("Installation cancelled.")

if __name__ == "__main__":
    main()