#!/bin/bash

# This script automates adding, committing, and pushing changes to git.

# Exit immediately if a command exits with a non-zero status.
set -e

# Add all changes to the staging area
echo "Adding all changes..."
git add .

# Ask for a commit message
read -p "Enter commit message: " commit_message

# If the commit message is empty, use a default message.
if [ -z "$commit_message" ]; then
  commit_message="Sync: $(date)"
  echo "No commit message entered. Using default: '$commit_message'"
fi

# Commit the changes
echo "Committing changes..."
git commit -m "$commit_message"

# Push the changes to the remote repository
echo "Pushing to remote..."
git push

echo "Git sync complete."
