# SIH-R2-Tigbits


# Team Tigbits submission for Smart India Hackathon:
# SIH Problem ID:1454
# Problem Statement Title:Create an intelligent system using AI/ML to detect phishing domains which imitate look and feel of genuine domains

# Video Submission:
## https://www.loom.com/share/7a0c1285ab2548439253603135469903?sid=3257df6f-2bc5-4029-862c-7c3cc7ed83dc

Organization NTRO
Category:Software

Application is created using Flask for making the backend API,and employs two ways user can enter his input:
1) URL
2) Snapshot of the website

The application is trained on a Logistic Regression Model and RESNet50 neural network for respective tasks.
Accuracy is ~96% in Logistic Regression and ~90% in RESNet50.


Setup with Python

```bash
python3 -m .venv venv
.venv/Scripts/activate
pip install -r requirements.txt
```
`Run application.py`
