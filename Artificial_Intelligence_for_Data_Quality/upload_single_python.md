# Upload a Single Python File to Bitbucket (Interactive Script)

This guide includes an interactive shell script to upload any single Python file from your local project to a Bitbucket repository.

---

## Interactive Shell Script

Save this as `upload_single_python.sh`:

```bash
#!/bin/bash

# ------------------------------
# Interactive Single File Upload to Bitbucket
# ------------------------------

# Prompt for Python file path (relative to project root)
read -p "Enter the relative path of the Python file to upload (e.g., jobs/job_customers.py): " FILE

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "❌ File '$FILE' does not exist!"
    exit 1
fi

# Prompt for Bitbucket workspace and repo
read -p "Enter Bitbucket workspace ID: " WORKSPACE
read -p "Enter Bitbucket repository name: " REPO
read -p "Enter branch name (default: main): " BRANCH
BRANCH=${BRANCH:-main}

REMOTE_URL="https://bitbucket.org/$WORKSPACE/$REPO.git"

# Initialize Git if needed
if [ ! -d ".git" ]; then
    git init
    echo "✅ Initialized new Git repository."
fi

# Optional: create basic .gitignore if not exists
if [ ! -f ".gitignore" ]; then
cat <<EOL > .gitignore
__pycache__/
*.pyc
.env
logs/
EOL
    echo "✅ Created .gitignore"
fi

# Add the file
git add "$FILE"

# Commit
git commit -m "Add single Python file: $FILE"

# Set branch name
git branch -M "$BRANCH"

# Add remote
git remote remove origin 2>/dev/null
git remote add origin "$REMOTE_URL"

# Push
git push -u origin "$BRANCH"

echo "✅ Successfully uploaded '$FILE' to Bitbucket repository: $REMOTE_URL on branch '$BRANCH'"
```

---

## Usage Instructions

1. Save the script in your project root.

```bash
chmod +x upload_single_python.sh
```

2. Run the script:

```bash
./upload_single_python.sh
```

3. Follow the prompts:

```
Enter the relative path of the Python file to upload (e.g., jobs/job_customers.py): jobs/job_customers.py
Enter Bitbucket workspace ID: myworkspace
Enter Bitbucket repository name: pyspark-project
Enter branch name (default: main):
```

The script will:

- Initialize Git if needed
- Create a basic `.gitignore` (if missing)
- Stage the specified Python file
- Commit it
- Add the remote repository
- Push to Bitbucket

---

### Notes

- This script works for **one file at a time**.  
- Future updates can be pushed with standard Git commands:

```bash
git add jobs/job_customers.py
git commit -m "Update job_customers.py"
git push
```

- Can be adapted to **SSH URLs** for passwordless pushes.
