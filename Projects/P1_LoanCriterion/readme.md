# Loan Criterion

A simple web application to determine loan eligibility using a pre-trained machine learning model. Users enter information on a web page, and the app predicts whether they qualify for a loan.

---
## Project Objective

This project is a comprehensive, hands-on introduction to full-stack web development using Python with Flask and MySQL with MongoDB. While it showcases the foundational components of building a web application, its primary goal is to demonstrate how a fully trained machine learning model can be integrated into a working web app and used as a core feature. The project leverages Python's pickling to serialise and load the trained model, enabling real-time predictions within the Flask application. This end-to-end example provides a template for deploying machine learning solutions in production-grade web environments.

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
## HINT: It's a brave new world of Agentic AI. Use GitHub Copilot in your Visual Studio Code to summarise the project and guide you to run, modify, improve, etc. 
## Future Improvements

- Dockerize the project for easier deployment (planned for version 2)
- Add more input features and improve the ML model
- Enhance front-end design

---

## License

[MIT](LICENSE)
