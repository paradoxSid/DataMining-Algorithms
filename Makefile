VIRTUAL_ENV=env
PYTHON_PATH=${VIRTUAL_ENV}/bin/python3
.DEFAULT_GOAL := help

venv:  ## Create python virtualenv at `./venv/`
	if [ ! -d "./${VIRTUAL_ENV}" ]; \
	then clear; virtualenv ${VIRTUAL_ENV}; \
	else clear; echo "Virtual Environment folder ${VIRTUAL_ENV} already exists"; \
	fi

install: venv  ## Install all required dependencies to run
	${PYTHON_PATH} -m pip install -r requirements.txt

kmeans: install ## Runs the kmeans algorithm for diffrent values of k and show a graph of k vs sse
	clear
	${PYTHON_PATH} clustring/k_means.py
	
dbscan: install ## Runs the dbscan algorithm for diffrent values of epsilon and min_count and print the values of same when number of clusters is same as a fixed k provided with minmum noise
	clear
	${PYTHON_PATH} clustring/db_scan.py

pca: install ## Runs the pca algorithm to transform the data set into 2 dimensional data
	clear
	${PYTHON_PATH} clustring/pca.py

em: install ## Runs the em algorithm after transforming the given dataset into 2-d for given value of k to plot the cluster obtained
	clear
	${PYTHON_PATH} clustring/em.py

denclue: install ## Runs the denclue algorithm after transforming the given dataset into 2 dimensional data for a given value of h, xi, epsilon
	clear
	${PYTHON_PATH} clustring/denclue.py

help:  ## Display this help
	# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'
.PHONY: help

