"# rv_analysis" 
Using emceeOmLMFITexe.py you can solve RVs to orbital parameters either from a given json samples file, or a simulated dataset given orbital parameters.
all parameters are defined in a file named params.json.
enjoy.


To connect an existing local directory to an existing remote Git repository using SSH, follow these steps:

---

### 1. **Navigate to the Local Directory**
   Open your terminal and navigate to the directory you want to connect:
   ```bash
   cd /path/to/your/local/directory
   ```

---

### 2. **Initialize Git (If Not Already Done)**
   If your local directory is not already a Git repository, initialize it:
   ```bash
   git init
   ```

---

### 3. **Add the Remote Repository via SSH**
   Find the SSH URL for your existing remote repository. It typically looks like:
   ```
   git@github.com:username/repository.git
   ```

   Add the remote repository to your local directory:
   ```bash
   git remote add origin git@github.com:username/repository.git
   ```

   Verify the remote was added:
   ```bash
   git remote -v
   ```
   This should display the SSH URL associated with the `origin`.

---

### 4. **Check SSH Access**
   Ensure you have SSH access to the remote repository. Test your SSH connection:
   ```bash
   ssh -T git@github.com
   ```
   If this is your first time connecting, you may need to add your SSH key to GitHub or another hosting service:
   - Generate an SSH key (if you don’t already have one):
     ```bash
     ssh-keygen -t ed25519 -C "your_email@example.com"
     ```
     (Replace with your email address.)
   - Add the public key to your hosting service (e.g., GitHub, GitLab, Bitbucket).

---

### 5. **Fetch Repository Data**
   Fetch data from the remote repository to ensure synchronization:
   ```bash
   git fetch origin
   ```

---

### 6. **Pull or Push Changes**
   - If the repository already contains commits, pull the latest changes:
     ```bash
     git pull origin main
     ```
     (Replace `main` with the branch name if different.)
   - If your local repository has changes to push, stage and commit them:
     ```bash
     git add .
     git commit -m "Initial commit from local directory"
     git push -u origin main
     ```

---

Your local directory is now connected to the remote Git repository via SSH! Let me know if you encounter any issues.
