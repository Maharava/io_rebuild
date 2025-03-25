#!/usr/bin/env python3
"""
Io Wake Word Project Structure Verification Script
Checks if all required files and directories are in place.
"""
import os
import sys
from pathlib import Path

# Terminal colors for supported platforms
if sys.platform != "win32" or os.environ.get("TERM") == "xterm":
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'
else:
    # No colors on unsupported platforms
    GREEN = RED = YELLOW = BLUE = BOLD = END = ''

def check_path(path, is_dir=False):
    """Check if a path exists and print result."""
    path_obj = Path(path)
    
    if (is_dir and path_obj.is_dir()) or (not is_dir and path_obj.is_file()):
        print(f"{GREEN}✓{END} {path}")
        return True
    else:
        print(f"{RED}✗{END} {path} {'directory' if is_dir else 'file'} not found")
        return False

def main():
    print(f"{BOLD}Io Wake Word Project Structure Verification{END}")
    print("=========================================")
    print(f"Running from: {os.getcwd()}")
    
    # Counters
    directories_found = 0
    directories_missing = 0
    files_found = 0
    files_missing = 0
    
    # Check if using nested structure
    if Path("src/io_wake_word").is_dir():
        print(f"{GREEN}✓{END} Using nested src layout (src/io_wake_word)")
        src_prefix = "src/io_wake_word"
    else:
        print(f"{YELLOW}!{END} Using flat src layout (src)")
        src_prefix = "src"
        print("Note: The nested structure is recommended for Python packages")
    
    if Path("apps/io_app").is_dir():
        print(f"{GREEN}✓{END} Using nested app layout (apps/io_app)")
        app_prefix = "apps/io_app"
    else:
        print(f"{YELLOW}!{END} Using flat app layout (apps)")
        app_prefix = "apps"
        print("Note: The nested structure is recommended for consistency")
    
    print()
    print(f"{BLUE}Checking root files:{END}")
    root_files = ["pyproject.toml", "README.md"]
    for file in root_files:
        if check_path(file):
            files_found += 1
        else:
            files_missing += 1
    
    print()
    print(f"{BLUE}Checking package structure:{END}")
    directories = [
        "src",
        "apps",
        f"{src_prefix}/audio",
        f"{src_prefix}/models",
        f"{src_prefix}/utils",
        f"{src_prefix}/diagnostics",
        f"{src_prefix}/cli",
        f"{app_prefix}/ui"
    ]
    for directory in directories:
        if check_path(directory, is_dir=True):
            directories_found += 1
        else:
            directories_missing += 1
    
    print()
    print(f"{BLUE}Checking core module files:{END}")
    core_files = [
        f"{src_prefix}/__init__.py",
        f"{src_prefix}/audio/__init__.py",
        f"{src_prefix}/audio/capture.py",
        f"{src_prefix}/audio/features.py",
        f"{src_prefix}/audio/vad.py",
        f"{src_prefix}/models/__init__.py",
        f"{src_prefix}/models/architecture.py",
        f"{src_prefix}/models/detector.py",
        f"{src_prefix}/models/trainer.py",
        f"{src_prefix}/utils/__init__.py",
        f"{src_prefix}/utils/actions.py",
        f"{src_prefix}/utils/config.py",
        f"{src_prefix}/utils/paths.py",
        f"{src_prefix}/diagnostics/__init__.py",
        f"{src_prefix}/diagnostics/analyzer.py",
        f"{src_prefix}/diagnostics/recovery.py",
        f"{src_prefix}/cli/__init__.py",
        f"{src_prefix}/cli/commands.py",
        f"{src_prefix}/cli/main.py",
    ]
    for file in core_files:
        if check_path(file):
            files_found += 1
        else:
            files_missing += 1
    
    print()
    print(f"{BLUE}Checking app files:{END}")
    app_files = [
        f"{app_prefix}/__init__.py",
        f"{app_prefix}/main.py",
        f"{app_prefix}/ui/__init__.py",
        f"{app_prefix}/ui/app.py",
        f"{app_prefix}/ui/config_panel.py",
        f"{app_prefix}/ui/training_panel.py",
    ]
    for file in app_files:
        if check_path(file):
            files_found += 1
        else:
            files_missing += 1
    
    print()
    print(f"{BLUE}Checking for old files that should be deleted:{END}")
    old_files = [
        "model_recovery.py",
        "recover_model.py",
        "data_analyzer.py",
        "main.py",
        "app.py",
        "config_panel.py",
        "training_panel.py",
        "setup.py",
        "io-wake-word.txt"
    ]
    
    files_to_delete = 0
    for file in old_files:
        if Path(file).is_file():
            print(f"{RED}Found old file: {file} - should be deleted{END}")
            files_to_delete += 1
    
    if files_to_delete == 0:
        print(f"{GREEN}No old files found that need deletion{END}")
    
    # Print summary
    print()
    print(f"{BLUE}{BOLD}Summary:{END}")
    print(f"Directories found: {GREEN}{directories_found}{END}")
    print(f"Directories missing: {RED}{directories_missing}{END}")
    print(f"Files found: {GREEN}{files_found}{END}")
    print(f"Files missing: {RED}{files_missing}{END}")
    print(f"Old files to delete: {RED}{files_to_delete}{END}")
    
    total_issues = directories_missing + files_missing + files_to_delete
    
    if total_issues == 0:
        print(f"\n{GREEN}{BOLD}All checks passed! Your project structure looks good.{END}")
    else:
        print(f"\n{YELLOW}{BOLD}Found {total_issues} issues to fix.{END}")
    
    print()
    print(f"{YELLOW}Next steps:{END}")
    print("1. Create any missing directories")
    print("2. Create any missing files")
    print("3. Delete any old files that are no longer needed")
    print("4. Make sure your imports use the correct package paths")
    print("   Example: from io_wake_word.audio import capture")
    
    return total_issues

if __name__ == "__main__":
    sys.exit(main())