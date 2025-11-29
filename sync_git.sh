#!/bin/bash

# This script automates adding, committing, pulling, and pushing changes to git.
# It includes checks for current branch, status, and prompts for confirmation.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Git Sync Script ---"

# Get current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $current_branch"
echo ""

# Show current status
echo "Checking git status..."
git status --short # Use --short for a concise view
echo ""

# Ask for confirmation to proceed
read -p "Review the changes above. Do you want to proceed with adding, committing, pulling, and pushing? (y/N): " confirm_proceed
if [[ ! "$confirm_proceed" =~ ^[Yy]$ ]]; then
  echo "Operation cancelled by user."
  exit 0
fi

# Add all changes to the staging area
echo "Adding all changes..."
git add .

# Ask for a commit message
read -p "Enter commit message (leave empty for default 'Sync: <date>'): " commit_message

# If the commit message is empty, use a default message.
if [ -z "$commit_message" ]; then
  commit_message="Sync: $(date +'%Y-%m-%d %H:%M:%S')"
  echo "No commit message entered. Using default: '$commit_message'"
fi

# Commit the changes
echo "Committing changes..."
git commit -m "$commit_message"

# Pull the latest changes from the remote before pushing
echo "Pulling latest changes from remote ($current_branch)..."
# Using --rebase to maintain a linear history. If conflicts occur, the script will exit.
git pull --rebase origin "$current_branch"

# Push the changes to the remote repository
echo "Pushing changes to remote ($current_branch)..."
git push origin "$current_branch"

echo "--- Git sync complete! ---"
