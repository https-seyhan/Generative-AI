# Interactive Script to Upload a Local File to Bitbucket

This guide provides a ready-to-run interactive shell script to upload **any file from your local drive** to a Bitbucket repository.

---

## Interactive Shell Script

Save this as `upload_local_file_bitbucket.sh`:

```bash
#!/bin/bash

# ------------------------------
# Interactive Local File Upload to Bitbucket
# ------------------------------

# Prompt for file path (relative to current folder)
read -p "Enter the path of the file to upload (e.g., jobs/job_customers.py): " FILE

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "❌ File '$FILE' does not exist!"
    exit 1
fi

# Prompt for Bitbucket info
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

# Create a basic .gitignore if not exists
if [ ! -f ".gitignore" ]; then
cat <<EOL > .gitignore
__pycache__/
*.pyc
.env
logs/
EOL
    echo "✅ Created .gitignore"
fi

# Stage the selected file
git add "$FILE"

# Commit the file
git commit -m "Add file: $FILE from local drive"

# Set branch
git branch -M "$BRANCH"

# Add remote
git remote remove origin 2>/dev/null
git remote add origin "$REMOTE_URL"

# Push to Bitbucket
git push -u origin "$BRANCH"

echo "✅ Successfully uploaded '$FILE' to Bitbucket repository: $REMOTE_URL on branch '$BRANCH'"
```

---

## How to Use

1. Save the script in your project folder.
2. Make it executable:

```bash
chmod +x upload_local_file_bitbucket.sh
```

3. Run the script:

```bash
./upload_local_file_bitbucket.sh
```

4. Follow the prompts:

```
Enter the path of the file to upload (e.g., jobs/job_customers.py): jobs/job_customers.py
Enter Bitbucket workspace ID: myworkspace
Enter Bitbucket repository name: pyspark-project
Enter branch name (default: main):
```

The script will:

- Initialize Git if not already initialized  
- Create a basic `.gitignore` if missing  
- Stage and commit the file  
- Add the Bitbucket remote repository  
- Push the file to the selected branch  

---

### Notes

- Works for **any single file**.  
- Future updates can be pushed with standard Git commands:

```bash
git add jobs/job_customers.py
git commit -m "Update job_customers.py"
git push
```

- Can be adapted to **SSH URLs** for passwordless pushes.
