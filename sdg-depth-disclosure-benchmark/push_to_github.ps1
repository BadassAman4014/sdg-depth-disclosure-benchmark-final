# ==============================================================================
# Automated Git LFS Setup & Push Script
# Project: Corporate SDG Disclosure Depth Scoring & Machine Learning Benchmarks
# ==============================================================================

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  🚀 Automated Git LFS Setup & GitHub Push Pipeline (Benchmark)" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan

# Step 1: Check Git installation
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/ and re-run." -ForegroundColor Yellow
    exit 1
}

# Step 2: Check & Initialize Git LFS
Write-Host "`n[1/6] Initializing Git Large File Storage (LFS)..." -ForegroundColor Yellow
git lfs install
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Warning: Git LFS might not be installed." -ForegroundColor Yellow
    Write-Host "If needed, install it via: winget install GitHub.GitLFS" -ForegroundColor Cyan
}

# Step 3: Check / Initialize Git Repo
Write-Host "`n[2/6] Checking Git Repository Status..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    Write-Host "Initializing new Git repository..." -ForegroundColor Cyan
    git init
    git branch -M main
} else {
    Write-Host "Git repository already initialized." -ForegroundColor Green
}

# Step 4: Ensure Git LFS is tracking large files
Write-Host "`n[3/6] Configuring Git LFS tracking rules..." -ForegroundColor Yellow
git lfs track "*.csv"
git lfs track "*.xlsx"
git lfs track "*.duckdb"
git lfs track "*.joblib"
git lfs track "*.parquet"

# Stage LFS config and gitignore
git add .gitattributes .gitignore

# Step 5: Check Remote Configuration
Write-Host "`n[4/6] Verifying GitHub Remote Origin..." -ForegroundColor Yellow
$currentRemote = git remote get-url origin 2>$null

if (-not $currentRemote) {
    Write-Host "No remote repository URL found." -ForegroundColor Yellow
    $repoUrl = Read-Host "Please enter your GitHub Repository URL (e.g., https://github.com/<username>/<repo>.git)"
    if ([string]::IsNullOrWhiteSpace($repoUrl)) {
        Write-Host "❌ Error: No repository URL provided. Aborting push." -ForegroundColor Red
        exit 1
    }
    git remote add origin $repoUrl.Trim()
    Write-Host "Added remote origin: $repoUrl" -ForegroundColor Green
} else {
    Write-Host "Found existing remote origin: $currentRemote" -ForegroundColor Green
    $changeRemote = Read-Host "Do you want to use this remote? (Y/n)"
    if ($changeRemote -match "^[Nn]") {
        $repoUrl = Read-Host "Enter new GitHub Repository URL"
        git remote set-url origin $repoUrl.Trim()
        Write-Host "Updated remote origin to: $repoUrl" -ForegroundColor Green
    }
}

# Step 6: Stage and Commit
Write-Host "`n[5/6] Staging files for commit (LFS objects will be indexed)..." -ForegroundColor Yellow
Write-Host "Note: Raw PDF reports are safely ignored via .gitignore to prevent uploading 20+ GB." -ForegroundColor Gray
git add .

$status = git status --porcelain
if ($status) {
    Write-Host "Creating commit..." -ForegroundColor Cyan
    git commit -m "feat: complete corporate SDG depth scoring engine, ML benchmarks, and filtered corpus with Git LFS"
} else {
    Write-Host "Working tree clean. Nothing new to commit." -ForegroundColor Green
}

# Step 7: Push to GitHub
Write-Host "`n[6/6] Pushing code and Large File Storage (LFS) objects to GitHub..." -ForegroundColor Yellow
Write-Host "Pushing to branch: main" -ForegroundColor Cyan
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=================================================================" -ForegroundColor Green
    Write-Host "  ✅ SUCCESS! Repository successfully pushed to GitHub with Git LFS." -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ Git push encountered an issue." -ForegroundColor Yellow
    Write-Host "If GitHub rejected because the remote repo has existing files (e.g. LICENSE/README):" -ForegroundColor Gray
    Write-Host "Run: git pull origin main --rebase" -ForegroundColor Cyan
    Write-Host "Then run: git push -u origin main" -ForegroundColor Cyan
}
