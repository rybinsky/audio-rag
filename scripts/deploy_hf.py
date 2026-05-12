#!/usr/bin/env python3
"""Deploy Audio RAG demo to Hugging Face Spaces.

This script prepares and deploys the Gradio demo to Hugging Face Spaces.

Usage:
    python scripts/deploy_hf.py --username YOUR_USERNAME --space-name audio-rag-demo --token YOUR_HF_TOKEN

Or with environment variables:
    export HF_USERNAME=your_username
    export HF_TOKEN=your_token
    python scripts/deploy_hf.py --space-name audio-rag-demo
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def copy_files(source_dir: Path, target_dir: Path) -> None:
    """Copy required files to target directory."""
    print(f"\n📁 Copying files to {target_dir}...")

    # Files to copy (source -> target)
    files_to_copy = [
        ("app.py", "app.py"),
        ("requirements_hf.txt", "requirements.txt"),
        ("README_HF.md", "README.md"),
        ("runtime.txt", "runtime.txt"),
    ]

    # Copy single files
    for src_name, dst_name in files_to_copy:
        src = source_dir / src_name
        dst = target_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✅ {src_name} → {dst_name}")
        else:
            print(f"  ⚠️  {src_name} not found, skipping")

    # Copy audio_rag package
    audio_rag_src = source_dir / "audio_rag"
    audio_rag_dst = target_dir / "audio_rag"
    if audio_rag_src.exists():
        if audio_rag_dst.exists():
            shutil.rmtree(audio_rag_dst)
        shutil.copytree(audio_rag_src, audio_rag_dst)
        print(f"  ✅ audio_rag/ → audio_rag/")
    else:
        raise FileNotFoundError("audio_rag/ directory not found!")

    # Copy config
    conf_src = source_dir / "conf" / "config_hf.yaml"
    conf_dst = target_dir / "conf" / "config_hf.yaml"
    if conf_src.exists():
        conf_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(conf_src, conf_dst)
        print(f"  ✅ conf/config_hf.yaml → conf/config_hf.yaml")
    else:
        raise FileNotFoundError("conf/config_hf.yaml not found!")

    print(f"\n✅ All files copied successfully!")


def init_git_repo(repo_dir: Path, username: str, space_name: str) -> None:
    """Initialize git repository and configure remote."""
    print(f"\n🔧 Initializing git repository...")

    # Check if already a git repo
    if (repo_dir / ".git").exists():
        print("  ⚠️  Git repository already exists, reinitializing...")
        shutil.rmtree(repo_dir / ".git")

    # Initialize git
    run_command(["git", "init"], cwd=repo_dir)
    print("  ✅ Git repository initialized")

    # Add remote
    remote_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    run_command(["git", "remote", "add", "origin", remote_url], cwd=repo_dir)
    print(f"  ✅ Remote added: {remote_url}")


def commit_and_push(repo_dir: Path, username: str, token: str, space_name: str) -> None:
    """Commit and push to Hugging Face."""
    print(f"\n📤 Committing and pushing to Hugging Face...")

    # Configure git user
    run_command(["git", "config", "user.email", f"{username}@users.noreply.huggingface.co"], cwd=repo_dir)
    run_command(["git", "config", "user.name", username], cwd=repo_dir)

    # Add all files
    run_command(["git", "add", "."], cwd=repo_dir)
    print("  ✅ Files staged")

    # Commit
    run_command(["git", "commit", "-m", "Deploy Audio RAG demo"], cwd=repo_dir)
    print("  ✅ Changes committed")

    # Push with token
    remote_url = f"https://{username}:{token}@huggingface.co/spaces/{username}/{space_name}"
    run_command(["git", "remote", "set-url", "origin", remote_url], cwd=repo_dir)

    # Push to main branch
    try:
        run_command(["git", "push", "-u", "origin", "main", "--force"], cwd=repo_dir)
        print("  ✅ Pushed to Hugging Face Spaces!")
    except subprocess.CalledProcessError as e:
        # Try master branch if main fails
        print("  ⚠️  Push to main failed, trying master branch...")
        try:
            run_command(["git", "push", "-u", "origin", "master", "--force"], cwd=repo_dir)
            print("  ✅ Pushed to Hugging Face Spaces (master branch)!")
        except subprocess.CalledProcessError as e2:
            print(f"  ❌ Failed to push: {e2.stderr}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Audio RAG demo to Hugging Face Spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With command line arguments
  python scripts/deploy_hf.py --username johndoe --space-name audio-rag-demo --token hf_xxx

  # With environment variables
  export HF_USERNAME=johndoe
  export HF_TOKEN=hf_xxx
  python scripts/deploy_hf.py --space-name audio-rag-demo

  # Dry run (prepare files but don't push)
  python scripts/deploy_hf.py --username johndoe --space-name audio-rag-demo --dry-run
        """
    )

    parser.add_argument(
        "--username",
        default=os.environ.get("HF_USERNAME"),
        help="Hugging Face username (or set HF_USERNAME env var)"
    )
    parser.add_argument(
        "--space-name",
        default="audio-rag-demo",
        help="Name of the Hugging Face Space (default: audio-rag-demo)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for deployment files (default: ./hf_deploy)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files but don't push to Hugging Face"
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip confirmation before pushing"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run:
        if not args.username:
            print("❌ Error: Username required (use --username or set HF_USERNAME)")
            sys.exit(1)
        if not args.token:
            print("❌ Error: Token required (use --token or set HF_TOKEN)")
            sys.exit(1)

    # Get source directory (project root)
    source_dir = Path(__file__).parent.parent.resolve()
    output_dir = Path(args.output_dir) if args.output_dir else source_dir / "hf_deploy"

    print("=" * 70)
    print("🎙️  Audio RAG - Hugging Face Spaces Deployment")
    print("=" * 70)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    if not args.dry_run:
        print(f"Hugging Face Space: {args.username}/{args.space_name}")
    print(f"Dry run: {args.dry_run}")

    # Create output directory
    if output_dir.exists():
        print(f"\n⚠️  Output directory already exists: {output_dir}")
        response = input("Remove and continue? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Aborted")
            sys.exit(0)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    try:
        copy_files(source_dir, output_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n" + "=" * 70)
        print("✅ Dry run completed successfully!")
        print(f"Files prepared in: {output_dir}")
        print("\nTo deploy manually:")
        print(f"  cd {output_dir}")
        print("  git init")
        print(f"  git remote add origin https://huggingface.co/spaces/{args.username or 'YOUR_USERNAME'}/{args.space_name}")
        print("  git add .")
        print("  git commit -m 'Deploy Audio RAG demo'")
        print("  git push -u origin main")
        print("=" * 70)
        sys.exit(0)

    # Confirm deployment
    if not args.skip_confirmation:
        print("\n" + "=" * 70)
        print(f"Ready to deploy to: https://huggingface.co/spaces/{args.username}/{args.space_name}")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Aborted")
            sys.exit(0)

    # Initialize git and push
    try:
        init_git_repo(output_dir, args.username, args.space_name)
        commit_and_push(output_dir, args.username, args.token, args.space_name)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Deployment failed!")
        print(f"Command: {e.cmd}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

    # Success!
    print("\n" + "=" * 70)
    print("🎉 Deployment successful!")
    print("=" * 70)
    print(f"Your demo is now live at:")
    print(f"  https://huggingface.co/spaces/{args.username}/{args.space_name}")
    print("\nIt may take a few minutes to build and start the demo.")
    print("Check the logs at:")
    print(f"  https://huggingface.co/spaces/{args.username}/{args.space_name}/logs")
    print("=" * 70)


if __name__ == "__main__":
    main()
