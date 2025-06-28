Write-Host "========================================" -ForegroundColor Green
Write-Host "PolicyPal - Push to GitHub" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Please follow these steps:" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com and sign in" -ForegroundColor White
Write-Host "2. Click the '+' icon and select 'New repository'" -ForegroundColor White
Write-Host "3. Name it: PolicyPal" -ForegroundColor White
Write-Host "4. Description: Enterprise RAG Chatbot for Policy Document Queries" -ForegroundColor White
Write-Host "5. Choose Public or Private" -ForegroundColor White
Write-Host "6. DO NOT initialize with README (we already have files)" -ForegroundColor White
Write-Host "7. Click 'Create repository'" -ForegroundColor White
Write-Host ""

$github_username = Read-Host "Enter your GitHub username"
Write-Host ""

Write-Host "Adding remote repository..." -ForegroundColor Cyan
git remote add origin "https://github.com/$github_username/PolicyPal.git"

Write-Host "Setting main branch..." -ForegroundColor Cyan
git branch -M main

Write-Host "Pushing code to GitHub..." -ForegroundColor Cyan
git push -u origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Success! Your code has been pushed to:" -ForegroundColor Green
Write-Host "https://github.com/$github_username/PolicyPal" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to continue" 