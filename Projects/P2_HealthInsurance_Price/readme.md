1. Data Ingestion
2. Data Transformation
3. Model Trainer
4. Model Evaluation
5. Model Deployment (on AWS)

CI/CD Pipelines with Github Actions
Python modules reference to read and learn:
1. setup
2. sys
3. custom exception handling
4. dataclass

Step 1: Set up project with git hub
Connect with your git hub repo with a readme and gitignore
Created src folder with "__init__.py"
Created setup.py
Created Requirements.txt, added -e . to connect it with setup.py to run automatically
Run pip install requirements -> installs requirements and automatically builds package from setup.py
A new package folder egg-info appears -> added this to gitignore.
Push it to all to repo.

Step 2: Structure, logging and exception
Created Components and pipeline folder with files for stages of ML trainings and pipelines.
Created Exception file with it's logic completed for custom exception
Created Logger file for logging
Created utils - used for
Pushed to git

Step 3: ML Project
Perform a quick a jupyter notebook experimentation with EDA and model training on the dataset- for insights. 
Apply this for data ingestion, transformation and model training modules.
To -do : Create a flwochart for logic flow for each modules - each module has classs and has method, parameters are udated and returned then passed to new module.

Step 4: Flask app
Create the predict pipeline - uses teh mdoel file and serve it to flask web app
Get form data from flask app and serve the prediction
To-do: Enhance the front end.

Step 5: Deployment and CI/CD
App deployment to Azure/AWS/free deployments - without docker
App deploymentt - with docker
Q: What is the difference? What happens when we use docker vs when we don't.
Q: What about training pipeline - The MLOPs implementation?
