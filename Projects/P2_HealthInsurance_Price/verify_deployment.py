#!/usr/bin/env python3
"""
Deployment Verification Script
This script helps verify that your deployment configuration is correct.
"""

import os
import sys
from pathlib import Path

def check_deployment_files():
    """Check if all necessary deployment files exist and are configured correctly."""
    
    print("🔍 Checking deployment configuration...")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Check if render.yaml exists and has correct content
    render_yaml = current_dir / "render.yaml"
    if render_yaml.exists():
        print("✅ render.yaml found")
        
        # Check if rootDir is correctly set
        with open(render_yaml, 'r') as f:
            content = f.read()
            if "rootDir: Projects/P2_HealthInsurance_Price" in content:
                print("✅ rootDir correctly set to Projects/P2_HealthInsurance_Price")
            else:
                print("❌ rootDir not found or incorrect in render.yaml")
                return False
    else:
        print("❌ render.yaml not found")
        return False
    
    # Check if requirements.txt has gunicorn
    requirements = current_dir / "requirements.txt"
    if requirements.exists():
        with open(requirements, 'r') as f:
            content = f.read()
            if "gunicorn" in content:
                print("✅ gunicorn found in requirements.txt")
            else:
                print("❌ gunicorn not found in requirements.txt")
                return False
    else:
        print("❌ requirements.txt not found")
        return False
    
    # Check if app.py has production settings
    app_py = current_dir / "app.py"
    if app_py.exists():
        with open(app_py, 'r') as f:
            content = f.read()
            if "os.environ.get('PORT'" in content and "debug=False" in content:
                print("✅ app.py configured for production")
            else:
                print("❌ app.py not configured for production")
                return False
    else:
        print("❌ app.py not found")
        return False
    
    # Check if GitHub Actions workflow exists
    workflow_dir = current_dir / ".github" / "workflows"
    if workflow_dir.exists():
        workflow_file = workflow_dir / "ci-cd.yml"
        if workflow_file.exists():
            print("✅ GitHub Actions workflow found")
        else:
            print("❌ GitHub Actions workflow not found")
            return False
    else:
        print("❌ .github/workflows directory not found")
        return False
    
    return True

def main():
    """Main verification function."""
    print("🚀 Health Insurance Price Predictor - Deployment Verification")
    print("=" * 60)
    
    if check_deployment_files():
        print("\n🎉 All deployment files are correctly configured!")
        print("\n📋 Next steps:")
        print("1. Push your changes to GitHub")
        print("2. Go to render.com and create a new Web Service")
        print("3. Connect your ML-Engineering-Projects repository")
        print("4. Render will automatically detect and use the render.yaml configuration")
        print("5. Your app will deploy from the Projects/P2_HealthInsurance_Price subfolder")
    else:
        print("\n❌ Deployment configuration has issues. Please fix them before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()
