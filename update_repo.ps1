# ==============================================================================
# Update Repo: Remove Golden Dataset Scripts & Push to GitHub
# ==============================================================================

Write-Host "Removing golden dataset creation and classification scripts..." -ForegroundColor Cyan

# 1. Remove golden dataset scripts from Git and filesystem
git rm -f scripts/create_golden_dataset.py 2>$null
git rm -f scripts/classify_golden_dataset.py 2>$null

Remove-Item -Path "scripts/create_golden_dataset.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "scripts/classify_golden_dataset.py" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "cleanup_repo.ps1" -Force -ErrorAction SilentlyContinue

# 2. Stage any remaining updates
git add -A

# 3. Commit
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m "chore: remove golden dataset creation and classification scripts"

# 4. Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "`n✅ Successfully removed golden dataset scripts and pushed changes to GitHub!" -ForegroundColor Green
