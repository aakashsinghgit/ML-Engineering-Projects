# Loan Criterion - Project Objective
This project is a **simple ML-powered web application** that predicts **loan approval** based on applicant details.

It serves as your **first step in ML Engineering**, demonstrating how to deploy a trained machine learning model using a Flask web app. Users enter information on a web page, and the app predicts whether they qualify for a loan.

While it showcases the foundational components of building a web application, its **primary goal is to demonstrate integrating a machine learning model as a core feature in a working web app**. This serves as a **template for deploying machine learning solutions in web environments**. Visit the main repo page for the full roadmap.

--- 
## Skills Covered

✅ Train a classification model to predict loan approval (yes/no).  
✅ Pickle the trained model for later inference.  
✅ Build a Flask web app (with MYSQL and MongoDB) to serve the model for user input and display predictions.

---
## Project Structure

```
P1_LoanCriterion/
├── .gitignore                # Specifies files and folders for Git to ignore
├── Flask_app_MLservice.docx  # Project documentation in Word format
├── Model_Building/        
│   ├── Model_Building_and_Pickling.ipynb  # Notebook for building & pickling the ML model
│   └── loan_approval_data.csv             # Dataset used for model training/testing
├── main.py                   # Main Flask application script
├── model.pkl                 # Serialised (pickled) trained ML model for inference
├── predict/                  
├── readme.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
└── templates/                # HTML templates for the Flask web app
    ├── home.html             # Landing/dashboard page
    ├── login.html            # User login page
    ├── register.html         # User registration page
    ├── stat_summary.html     # Displays statistical summaries
    └── summary_report.html   # Shows summarised reports/results
```

---

## Features

- Collects user input via a web form
- Uses a pre-trained ML model (pickle file) for predictions
- Returns instant feedback ("Eligible"/"Not Eligible") on the web page

---

## Prerequisites and Setup

1. **Install MySQL Server**  
   Make sure MySQL Server is installed and running on your machine.  
   > The application connects to MySQL using the `root` user and expects the password you provide in your MySQL installation.   
   > **Note:** The MySQL root password used in the app must match the password set in `main.py` (see line 18; update this line with your actual root password if needed).
   > It will appear as below:
   ```
   app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:yourpass@localhost:3306/login_store'
   # replace 'yourpass' with your actual MySQL root password
   ```

2. **Install Python**  
   Python 3.x must be installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

3. **Create and Activate a Virtual Environment**  
   It's recommended to use a virtual environment for package management:
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

4. **Install Python Dependencies**  
   Once the virtual environment is activated, install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
5. **Ensure the model file exists**

Ensure `model.pkl` is inside the `root/` directory as shown in Project Structure.   
You can train a new model using 'Model_Building/' contents, which creates a model.pkl file on run. Export it to the 'root/' directory.

6. **Run the Application**  
   ```sh
   python main.py
   ```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

---
💡 Tip: We are in the brave new world of Agentic AI. Use GitHub Copilot in VS Code to summarize, modify, improve, and guide you while working on this project.

📈 **Next Step in Your Learning Path:**

Continue your ML Engineering journey with modular structuring, CI/CD pipelines, and containerization in:

[🔗 Project 2: Health Insurance Price Predictor](https://github.com/aakashsinghgit/ML-Engineering-Projects/tree/main/Projects/P2_HealthInsurance_Price)
---
## 🏗️ Contribution

Feel free to fork, open issues, or submit pull requests for improvements or new project ideas!

## 📄 License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*Maintained by [aakashsinghgit](https://github.com/aakashsinghgit)*


