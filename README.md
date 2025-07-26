# The Selection Vector - Competition Toolkit

This repository contains the official toolkit for all participants of the **The Selection Vector** machine learning competition, hosted by AIRAC. The tools provided here are designed to help you validate your submissions locally and ensure your environment matches our official evaluation server.

## Getting Started

To ensure a smooth and fair competition, all participants **must** use the files in this repository to set up their environment and test their submissions.



### 1. Set Up Your Python Environment

It is highly recommended to use a virtual environment to avoid conflicts with your existing Python packages.

```bash
# Create a virtual environment
python -m venv venv
```

### 2. Install Required Libraries

Install the exact versions of all required libraries using the provided `requirements.txt` file. This step is mandatory to prevent version conflicts.

```bash
pip install -r requirements.txt
```

## How to Validate Your Submission Locally

Before you submit your `.pkl` file, you should use the `run_evaluation.py` script to ensure it can be loaded and run without errors.

### 1. Set Up the Test Directory Structure

The evaluation script requires a specific folder structure to work. Create the following folders in your project directory:

```
your_project_folder/
│
├── day1_submissions/
│   └── YourFirstName_YourLastName.pkl  <-- Place your submission here
│
├── day1_validation/
│   ├── X_val.csv                     <-- Create a dummy CSV file
│   └── y_val.csv                     <-- Create another dummy CSV file
│
└── run_evaluation.py                 <-- The script from this repo
```

**Note:** You only need to create dummy `X_val.csv` and `y_val.csv` files for the test to run; they can contain any placeholder data. The script will not produce a real score, but it will confirm if your pipeline is valid.

### 2. Run the Evaluation Script

Open your terminal, make sure your virtual environment is activated, and run the script, pointing it to the correct day:

```bash
python run_evaluation.py --day 1
```

### 3. Check the Output

* If you see a **`[SUCCESS]`** message, your pipeline is valid and ready for submission.
* If you see a **`[FAILED]`** message, the output will contain an error description. Debug your pipeline before submitting.

## Custom Classes

If you create a custom transformer class (e.g., a custom encoder), you must submit the class code to the organizers for review. Once approved, the class will be added to the `custom_definitions.py` file in this repository. Please `git pull` regularly to ensure you have the latest version of this file.

---
Good luck, and may the best model win!
