#!/usr/bin/env python3
import os
import subprocess
import time
import argparse
import shutil
from pathlib import Path

# Config
WORKSPACE_ROOT = Path(".agent/workspaces")
TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

def get_git_root():
    try:
        res = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        return Path(res.stdout.strip())
    except subprocess.CalledProcessError:
        return Path(os.getcwd())

def gc(verbose=False):
    """Remove worktrees older than 7 days."""
    if not WORKSPACE_ROOT.exists():
        if verbose: print(f"No workspace root found at {WORKSPACE_ROOT}")
        return

    now = time.time()
    for item in WORKSPACE_ROOT.iterdir():
        if not item.is_dir(): continue
        
        mtime = item.stat().st_mtime
        age = now - mtime
        
        if age > TTL_SECONDS:
            print(f"🧹 Garbage Collecting old workspace: {item.name} (Age: {age/86400:.1f} days)")
            try:
                # 1. Force remove from git
                subprocess.run(["git", "worktree", "remove", "--force", str(item)], check=False)
                # 2. Delete physical folder if it still exists (git remove --force should handle it, but being sure)
                if item.exists():
                    shutil.rmtree(item)
                # 3. Clean up broken branches if any
                branch_name = f"session/{item.name}"
                subprocess.run(["git", "branch", "-D", branch_name], check=False)
            except Exception as e:
                print(f"⚠️ Error cleaning up {item.name}: {e}")

    # Final git cleanup for any dangling worktrees not in the folder
    subprocess.run(["git", "worktree", "prune"], check=False)

def setup(name):
    """Create a new worktree for an agent session."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(os.environ.get("ANTIGRAVITY_TIMESTAMP", time.time()))
    session_id = f"{name}_{timestamp}"
    path = WORKSPACE_ROOT / session_id
    branch = f"session/{session_id}"

    print(f"🚀 Setting up isolated workspace: {session_id}")
    
    try:
        # Create worktree with a new branch
        subprocess.run(["git", "worktree", "add", "-b", branch, str(path)], check=True)
        print(f"✅ Created worktree at {path} on branch {branch}")
        return str(path)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create worktree: {e}")
        return None

def cleanup(path_str):
    """Manually cleanup a specific workspace."""
    path = Path(path_str)
    if not path.exists():
        print(f"⚠️ Path {path} does not exist.")
        return

    print(f"🧹 Cleaning up workspace: {path.name}")
    try:
        subprocess.run(["git", "worktree", "remove", "--force", str(path)], check=True)
        # We don't delete the branch here automatically, usually orchestrator merges it first.
        # But we can prune.
        subprocess.run(["git", "worktree", "prune"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description="Antigravity Workspace Manager (Git Worktree)")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # GC command
    subparsers.add_parser("gc", help="Run garbage collection on old workspaces")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Create a new isolated workspace")
    setup_parser.add_argument("name", help="Name of the session")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove a specific workspace")
    cleanup_parser.add_argument("path", help="Path to the workspace")

    args = parser.parse_args()

    if args.command == "gc":
        gc(verbose=True)
    elif args.command == "setup":
        path = setup(args.name)
        if path: print(f"PATH={path}") # For orchestration capture
    elif args.command == "cleanup":
        cleanup(args.path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
