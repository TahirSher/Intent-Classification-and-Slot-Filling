
import os
import sys

def fix_file_encoding(filepath):
    """Fix encoding for a single file"""
    try:
        # Try to read with utf-8
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"OK: {filepath}")
        return True
    except UnicodeDecodeError:
        print(f"FIXING: {filepath}")
        try:
            # Try reading with latin-1 and writing as utf-8
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"FIXED: {filepath}")
            return True
        except Exception as e:
            print(f"ERROR: {filepath} - {e}")
            return False

def main():
    """Fix all Python files in current directory"""
    current_dir = os.getcwd()
    print(f"Fixing Python files in: {current_dir}\n")
    
    python_files = [f for f in os.listdir(current_dir) if f.endswith('.py')]
    
    fixed_count = 0
    error_count = 0
    
    for pyfile in python_files:
        filepath = os.path.join(current_dir, pyfile)
        if fix_file_encoding(filepath):
            fixed_count += 1
        else:
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Total files: {len(python_files)}")
    print(f"OK/Fixed: {fixed_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()