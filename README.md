
# Generic DS Projects

This repo contains a sample data science project. The goal is not simply to model a problem but to also tackle some common things that come up in a real job. These include:

1. APIs
2. Testing
3. Continuous integration and Continuous deployment

[![CircleCI](https://circleci.com/gh/dtsukiyama/Generic-DS-Projects.svg?style=svg)](https://circleci.com/gh/dtsukiyama/Generic-DS-Projects)

# Contents

1. case-study.ipynb
2. utils
  - models.py
  - pipeline.py
  - utils.py
  - models/
3. api.py
4. local_test.sh
5. README.md
6. requirements.txt
7. vehicles_booking_history.csv
8. Dockerfile
9. CircleCI config

# Setup

Create a virtual environment (virtualenv of Anaconda), install requirements.

```
virtualenv -p python3 env
source env/bin/activate

pip install -r requirements.txt
```

# Case Study

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
