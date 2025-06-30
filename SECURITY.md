# Security Best Practices

## 🔒 Never Commit Secrets

**IMPORTANT**: Never commit API keys, passwords, tokens, or any other sensitive information to this repository.

### What NOT to commit:
- API keys (OpenAI, Google, etc.)
- SSH private keys
- Passwords
- Access tokens
- Database credentials
- Environment files (.env)
- Configuration files with secrets

### How to handle secrets:

1. **Use Environment Variables**: Store secrets in environment variables
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Use .env files locally** (already in .gitignore):
   ```bash
   # .env file (never commit this)
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Use GitHub Secrets** for CI/CD:
   - Go to your repository settings
   - Navigate to Secrets and variables → Actions
   - Add your secrets there

### If you accidentally commit a secret:

1. **Immediately revoke the secret** (change API key, etc.)
2. **Remove from git history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch path/to/file/with/secret' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** to remove from remote:
   ```bash
   git push origin --force --all
   ```

### Security Checklist:
- [ ] No API keys in code
- [ ] No passwords in code
- [ ] No SSH keys in repository
- [ ] .env files are in .gitignore
- [ ] All secrets use environment variables
- [ ] CI/CD uses GitHub Secrets

### Reporting Security Issues:
If you find a security vulnerability, please report it privately to the repository maintainer. 