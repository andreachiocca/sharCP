# Publishing on GitHub

GitHub repository: https://github.com/andreachiocca/sharCP.

## Create and upload

1. Open https://github.com/new while signed into the relevant account.
2. Create the repository `sharCP` and choose **Public** if the code is approved for release. **Do not initialise** it with a README, Gitignore, or licence, since the package already includes these files.
3. Extract `sharCP_github_ready.zip`. From the extracted `sharCP` directory:

```bash
git init -b main
git add .
git commit -m "Initial public ECP sharp-notch fatigue research release"
git remote add origin https://github.com/andreachiocca/sharCP.git
git push -u origin main
```

4. Check the original calibration parameters and generated master curves before claiming exact reproduction of the manuscript results. Check licensing and access restrictions for all experimental data before public release.
5. Add the journal's formal citation and DOI to `CITATION.cff` when available.

The full original FE archives should be retained as separate research assets, not pushed into Git history. The supplied prediction archive contains at least one FE CSV exceeding the normal 100 MB GitHub file limit. It can be deposited in Zenodo or distributed separately, subject to rights and access restrictions.
