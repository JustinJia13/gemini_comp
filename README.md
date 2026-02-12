# gemini_comp

Trading competition project using AWS S3 for data storage.

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd gemini_comp

# 2. Run the setup script
bash setup.sh

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Access Key, region (us-east-1), and output format (json)

# 5. Launch Jupyter
jupyter notebook
```

## Project Structure

```
gemini_comp/
├── aws_demo.ipynb   # AWS/S3 tutorial and usage examples
├── setup.sh         # Environment setup script
└── README.md
```
