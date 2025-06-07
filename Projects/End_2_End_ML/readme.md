1. Data Ingestion
2. Data Transformation
3. Model Trainer
4. Model Evaluation
5. Model Deployment (on AWS)

CI/CD Pipelines with Github Actions

Step 1: Set up project with git hub
Connect with your git hub repo with a readme and gitignore
Created src folder with "__init__.py"
Created setup.py
Created Requirements.txt, added -e . to connect it with setup.py to run automatically
Run pip install requirements -> installs requirements and automatically builds package from setup.py
A new package folder egg-info appears -> added this to gitignore.
Push it to all to repo.

Step 2: Structure, logging and exception


Python modules reference to read and learn:
1. setup
2. sys
3. custom exception handling