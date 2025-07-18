"""
🚀 Employee Salary Prediction - Project Launcher

This script helps you get started with the Employee Salary Prediction project.
It will guide you through the entire machine learning pipeline step by step.
"""

import os
import sys
import subprocess
import time

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {text}")
    print("="*60)

def print_step(step_num, title, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {title}")
    print(f"   {description}")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully!")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"[ERROR] Error in {description}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] Error running command: {e}")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Main launcher function"""
    print_header("Employee Salary Prediction - ML Project Launcher")
    
    print("""
    Welcome to the Employee Salary Prediction Machine Learning Project! 🎉
    
    This launcher will guide you through the complete ML pipeline:
    1. Install dependencies
    2. Generate sample dataset
    3. Preprocess the data
    4. Train machine learning models
    5. Evaluate model performance
    6. Make salary predictions
    
    Perfect for beginners learning machine learning! 📚
    """)
    
    # Check if user wants to continue
    response = input("\n🚀 Ready to start your ML journey? (y/n): ").lower().strip()
    if response != 'y':
        print("👋 Maybe next time! Feel free to explore the files manually.")
        return
    
    # Step 1: Check and install dependencies
    print_step(1, "Check Dependencies", "Verifying required Python packages")
    
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        install = input("Install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            if not run_command("pip install -r requirements.txt", "Installing dependencies"):
                print("[ERROR] Failed to install dependencies. Please install manually.")
                return
        else:
            print("[WARNING] Some features may not work without required packages.")
    else:
        print("[OK] All required packages are installed!")
    
    # Step 2: Generate dataset
    print_step(2, "Generate Dataset", "Creating sample employee salary data")
    if not run_command("python src/create_dataset.py", "Generating sample dataset"):
        print("❌ Failed to generate dataset. Check if Python is installed correctly.")
        return
    
    # Step 3: Data preprocessing
    print_step(3, "Preprocess Data", "Cleaning and preparing data for machine learning")
    if not run_command("python src/data_preprocessing.py", "Preprocessing data"):
        print("❌ Failed to preprocess data. Check the error messages above.")
        return
    
    # Step 4: Train models
    print_step(4, "Train Models", "Training multiple ML algorithms and comparing performance")
    print("⏳ This step may take a few minutes...")
    if not run_command("python src/model_training.py", "Training machine learning models"):
        print("❌ Failed to train models. Check the error messages above.")
        return
    
    # Step 5: Evaluate models
    print_step(5, "Evaluate Models", "Creating visualizations and performance reports")
    if not run_command("python src/model_evaluation.py", "Evaluating models"):
        print("❌ Failed to evaluate models. Check the error messages above.")
        return
    
    # Success message
    print_header("🎉 SUCCESS! Your ML Project is Ready!")
    
    print("""
    Congratulations! You've successfully completed the machine learning pipeline! 🎊
    
    📊 What you've accomplished:
    ✅ Generated a realistic employee salary dataset
    ✅ Preprocessed and cleaned the data
    ✅ Trained 8 different machine learning models
    ✅ Evaluated and compared model performance
    ✅ Created visualizations and reports
    
    🎯 Next Steps:
    """)
    
    # Offer next steps
    while True:
        print("""
    Choose what you'd like to do next:
    
    1. 🎯 Make salary predictions (interactive)
    2. 📓 Open Jupyter notebook for data exploration
    3. 📁 View generated files and reports
    4. 📚 Read the beginner's guide
    5. 🔍 Check model performance summary
    6. 🚪 Exit
    """)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\n🎯 Starting interactive salary prediction...")
            run_command("python src/prediction.py", "Loading prediction interface")
            
        elif choice == '2':
            print("\n📓 Opening Jupyter notebook...")
            print("Note: This will open in your web browser")
            run_command("jupyter notebook notebooks/01_exploratory_data_analysis.ipynb", 
                       "Starting Jupyter notebook")
            
        elif choice == '3':
            print("\n📁 Generated files:")
            folders_to_check = ['data', 'models']
            for folder in folders_to_check:
                if os.path.exists(folder):
                    files = os.listdir(folder)
                    print(f"\n📂 {folder}/")
                    for file in files:
                        print(f"   📄 {file}")
                        
        elif choice == '4':
            print("\n📚 Opening beginner's guide...")
            if os.path.exists('BEGINNER_GUIDE.md'):
                print("📖 Check BEGINNER_GUIDE.md for detailed explanations!")
            else:
                print("❌ Beginner's guide not found.")
                
        elif choice == '5':
            print("\n🔍 Model Performance Summary:")
            if os.path.exists('models'):
                print("✅ Models trained and saved in 'models/' folder")
                print("✅ Check the terminal output above for performance metrics")
                print("✅ Visualizations saved as PNG files in 'models/' folder")
            else:
                print("❌ Models folder not found. Run the training step again.")
                
        elif choice == '6':
            print("\n👋 Thank you for using the Employee Salary Prediction project!")
            print("🎓 Keep practicing and exploring machine learning!")
            break
            
        else:
            print("❌ Invalid choice. Please enter a number between 1-6.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Project launcher interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("💡 Try running the individual scripts manually.")
