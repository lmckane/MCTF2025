#!/usr/bin/env python
"""
Migration script to move tests from old test directories to the new consolidated structure.

This script helps with the transition from the old test directories (hrl/tests and hrl/test)
to the new consolidated testing framework in hrl/testing.
"""

import os
import shutil
import argparse
from pathlib import Path

# Define source and destination directories
SOURCE_DIRS = {
    'hrl/tests': 'environment', 
    'hrl/test': 'components'
}

# File mapping for special cases
FILE_MAPPING = {
    'hrl/tests/test_team_coordination.py': 'hrl/testing/environment/test_team_coordination.py',
    'hrl/tests/test_env.py': 'hrl/testing/environment/test_env.py',
    'hrl/test/test_hrl.py': 'hrl/testing/components/test_hrl.py',
    'hrl/test/run_tests.py': 'hrl/testing/components/run_tests.py',
}

def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def migrate_file(src_path, dest_path, dry_run=False):
    """Migrate a file from source to destination."""
    if os.path.exists(dest_path):
        print(f"SKIP: File already exists at destination: {dest_path}")
        return False
    
    if dry_run:
        print(f"WOULD COPY: {src_path} -> {dest_path}")
    else:
        print(f"COPYING: {src_path} -> {dest_path}")
        dest_dir = os.path.dirname(dest_path)
        ensure_dir_exists(dest_dir)
        shutil.copy2(src_path, dest_path)
    return True

def migrate_tests(dry_run=False):
    """Migrate test files from old to new structure."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cwd = os.getcwd()
    
    # For compatibility with different working directories
    if not os.path.exists('hrl') and os.path.exists(os.path.join(repo_root, 'hrl')):
        os.chdir(repo_root)
    
    # Ensure target directories exist
    for target_dir in ['hrl/testing/environment', 'hrl/testing/components', 'hrl/testing/opponents', 'hrl/testing/utils']:
        ensure_dir_exists(target_dir)
    
    # Track migration results
    migrated = []
    skipped = []
    
    # First, handle special cases
    for src_path, dest_path in FILE_MAPPING.items():
        if os.path.exists(src_path):
            success = migrate_file(src_path, dest_path, dry_run)
            if success:
                migrated.append((src_path, dest_path))
            else:
                skipped.append((src_path, dest_path))
    
    # Then handle any remaining files
    for src_dir, target_subdir in SOURCE_DIRS.items():
        if not os.path.exists(src_dir):
            continue
            
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            
            # Skip if not a Python file or already handled
            if not src_path.endswith('.py') or src_path in FILE_MAPPING:
                continue
                
            # Determine destination
            dest_path = f'hrl/testing/{target_subdir}/{filename}'
            
            # Migrate the file
            success = migrate_file(src_path, dest_path, dry_run)
            if success:
                migrated.append((src_path, dest_path))
            else:
                skipped.append((src_path, dest_path))
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Migration Summary ({'DRY RUN - no files were copied' if dry_run else 'MIGRATION COMPLETE'})")
    print("=" * 70)
    print(f"Files migrated: {len(migrated)}")
    print(f"Files skipped (already exist): {len(skipped)}")
    
    if migrated:
        print("\nMigrated files:")
        for src, dest in migrated:
            print(f"  {src} -> {dest}")
    
    # Restore working directory
    os.chdir(cwd)
    
    return migrated, skipped

def main():
    """Main function to run migration script."""
    parser = argparse.ArgumentParser(description='Migrate tests to new structure')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be migrated without copying files')
    args = parser.parse_args()
    
    migrate_tests(dry_run=args.dry_run)
    
    if not args.dry_run:
        print("\nMigration completed. You can now use the new testing framework:")
        print("  python -m hrl.testing.run_all_tests --all")
        print("  python -m hrl.testing.run_all_tests --components")
        print("  python -m hrl.testing.run_all_tests --environment")
        print("  python -m hrl.testing.run_all_tests --opponents")
    else:
        print("\nThis was a dry run. No files were copied.")
        print("Run without --dry-run to perform the actual migration.")

if __name__ == "__main__":
    main() 