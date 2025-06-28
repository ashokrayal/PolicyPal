@echo off
echo ========================================
echo PolicyPal - Push to GitHub
echo ========================================
echo.

echo Please follow these steps:
echo 1. Go to https://github.com and sign in
echo 2. Click the "+" icon and select "New repository"
echo 3. Name it: PolicyPal
echo 4. Description: Enterprise RAG Chatbot for Policy Document Queries
echo 5. Choose Public or Private
echo 6. DO NOT initialize with README (we already have files)
echo 7. Click "Create repository"
echo.

set /p github_username="Enter your GitHub username: "
echo.

echo Adding remote repository...
git remote add origin https://github.com/%github_username%/PolicyPal.git

echo Setting main branch...
git branch -M main

echo Pushing code to GitHub...
git push -u origin main

echo.
echo ========================================
echo Success! Your code has been pushed to:
echo https://github.com/%github_username%/PolicyPal
echo ========================================
echo.

pause 