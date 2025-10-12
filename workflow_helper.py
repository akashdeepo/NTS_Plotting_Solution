"""
Simple workflow helper for managing the project
Run this to help with common tasks
"""

import os
import shutil
from datetime import datetime
import json
import sys

def status():
    """Check current project status"""
    print("="*60)
    print("PROJECT STATUS")
    print("="*60)

    # Check current/ folder
    current_files = []
    if os.path.exists('current'):
        for root, dirs, files in os.walk('current'):
            current_files.extend(files)

    print(f"\n[current/] - Work in Progress:")
    if current_files:
        print(f"   {len(current_files)} files present")
        for f in current_files[:5]:  # Show first 5
            print(f"   - {f}")
        if len(current_files) > 5:
            print(f"   ... and {len(current_files)-5} more")
    else:
        print("   Empty (ready for new work)")

    # Check latest_commits/
    latest_files = []
    if os.path.exists('latest_commits'):
        for root, dirs, files in os.walk('latest_commits'):
            if '.pdf' in root or files:
                latest_files.extend(files)

    print(f"\n[latest_commits/] - Ready to Commit:")
    print(f"   {len([f for f in latest_files if f.endswith('.py')])} Python files")
    print(f"   {len([f for f in latest_files if f.endswith('.pdf')])} PDF figures")

    # Check git status
    print(f"\n[Git Status]:")
    os.system("git status --short")

def prepare_commit():
    """Prepare files for commit by moving from current/ to latest_commits/"""
    print("="*60)
    print("PREPARING FOR COMMIT")
    print("="*60)

    if not os.path.exists('current'):
        print("[ERROR] No current/ folder found")
        return

    # Find Python files in current/
    py_files = []
    for root, dirs, files in os.walk('current'):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))

    if py_files:
        print("\nFound Python files:")
        for i, f in enumerate(py_files):
            print(f"  {i+1}. {f}")

        choice = input("\nWhich file is the main code? (number): ")
        try:
            main_file = py_files[int(choice)-1]
            dest = 'latest_commits/generate_figures.py'
            shutil.copy2(main_file, dest)
            print(f"[OK] Copied {main_file} -> {dest}")
        except:
            print("[ERROR] Invalid choice")
            return

    # Find PDF figures
    pdf_count = 0
    for root, dirs, files in os.walk('current'):
        for file in files:
            if file.endswith('.pdf') and 'Figure' in file:
                src = os.path.join(root, file)
                dst = os.path.join('latest_commits/figures', file)
                shutil.copy2(src, dst)
                pdf_count += 1

    print(f"[OK] Copied {pdf_count} PDF figures to latest_commits/figures/")

    # Update parameters.json with timestamp
    params_file = 'latest_commits/parameters.json'
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        params['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        print("[OK] Updated parameters.json")

    print("\n[SUCCESS] Ready to commit! Run:")
    print("   git add latest_commits/ README.md")
    print("   git commit -m 'Update figures and code'")

def archive_current():
    """Archive current/ folder with timestamp"""
    print("="*60)
    print("ARCHIVING CURRENT WORK")
    print("="*60)

    if not os.path.exists('current') or not os.listdir('current'):
        print("[ERROR] Nothing to archive in current/")
        return

    # Create archive name with date and description
    desc = input("Brief description (e.g., 'case3_implementation'): ")
    date_str = datetime.now().strftime('%Y-%m-%d')
    archive_name = f"archive/{date_str}_{desc}"

    # Move current to archive
    shutil.move('current', archive_name)
    print(f"[OK] Archived to: {archive_name}")

    # Create new empty current/
    os.makedirs('current/manuscripts', exist_ok=True)
    print("✓ Created fresh current/ folder")

def new_task():
    """Set up for new task from Dr. Rachev"""
    print("="*60)
    print("SETTING UP NEW TASK")
    print("="*60)

    # Check if current/ is empty
    if os.path.exists('current') and os.listdir('current'):
        print("[WARNING]  current/ folder not empty!")
        choice = input("Archive it first? (y/n): ")
        if choice.lower() == 'y':
            archive_current()

    # Create structure
    os.makedirs('current/manuscripts', exist_ok=True)
    os.makedirs('current/figures', exist_ok=True)

    print("\n[SUCCESS] Ready for new task!")
    print("\nNext steps:")
    print("1. Copy Dr. Rachev's manuscript to current/manuscripts/")
    print("2. Save email as current/instructions.txt")
    print("3. Start coding in current/working_code.py")

def clean():
    """Clean up temporary files"""
    print("="*60)
    print("CLEANING TEMPORARY FILES")
    print("="*60)

    patterns = ['*.pyc', '__pycache__', '*.tmp', '*.bak', '.ipynb_checkpoints']
    count = 0

    for pattern in patterns:
        for root, dirs, files in os.walk('.'):
            if 'archive' in root or 'private' in root:
                continue
            for item in dirs + files:
                if pattern.replace('*', '') in item:
                    path = os.path.join(root, item)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    count += 1
                    print(f"  Removed: {path}")

    print(f"\n✓ Cleaned {count} items")

def main():
    """Main menu"""

    commands = {
        '1': ('status', 'Check project status'),
        '2': ('new_task', 'Set up new task from Dr. Rachev'),
        '3': ('prepare_commit', 'Move files from current/ to latest_commits/'),
        '4': ('archive_current', 'Archive current work'),
        '5': ('clean', 'Clean temporary files'),
        'q': (None, 'Quit')
    }

    while True:
        print("\n" + "="*60)
        print("WORKFLOW HELPER")
        print("="*60)

        for key, (_, desc) in commands.items():
            print(f"  {key}. {desc}")

        choice = input("\nChoice: ").strip()

        if choice == 'q':
            break
        elif choice in commands and commands[choice][0]:
            print()
            globals()[commands[choice][0]]()
        else:
            print("Invalid choice")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow direct command: python workflow_helper.py status
        cmd = sys.argv[1]
        if cmd in ['status', 'new_task', 'prepare_commit', 'archive_current', 'clean']:
            globals()[cmd]()
    else:
        main()