
# Generic DS Projects

[![CircleCI](https://circleci.com/gh/dtsukiyama/Generic-DS-Projects.svg?style=svg)](https://circleci.com/gh/dtsukiyama/Generic-DS-Projects)

This repo contains a sample data science project. The goal is not simply to model a problem but to also tackle some common things that come up in a real job. These include:

1. APIs
2. Testing
3. Continuous integration and Continuous deployment
4. Design Docs


# Contents

1. case-study-optimzation.ipynb
2. case-study-fraud.ipynb
3. utils
  - models.py
  - pipeline.py
  - utils.py
  - models/
4. api.py
5. local_test.sh
6. README.md
7. requirements.txt
8. vehicles_booking_history.csv
9. Dockerfile
10. CircleCI config

# Setup

Create a virtual environment (virtualenv of Anaconda), install requirements.

```
virtualenv -p python3 env
source env/bin/activate

pip install -r requirements.txt
```

# Case Studies

Open up the Jupyter notebook:

```
jupyter notebook
```

# API

From the main directory (where api.py is) run:

```
python api.py
```

From another terminal you can run a test which will deliver a payload to the endpoint:

```
./local_test.sh
```

If for some reason the executable is not executable (ha), run:

```
chmod +x local_test.sh
```
