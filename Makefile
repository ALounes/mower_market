TAG="\n\n\033[0;32m\#\#\# "
END=" \#\#\# \033[0m\n"

init:
	@echo $(TAG)Install prod requirements$(END)
	pipenv install

	@echo $(TAG)Install dev requirements$(END)
	pipenv install --dev

lint:
	@echo $(TAG)Check code quality$(END)
	pipenv run pylint src/

unittest:
	@echo $(TAG)test(END)
	pipenv run pytest --cov=coedi --cov-report html --cov-report xml --junitxml=test_report.xml tests/ -vv
