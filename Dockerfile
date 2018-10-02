FROM ubuntu:17.10

# ------------------------#
# Install Open JDK & Git  #
# ------------------------#

RUN	apt-get -y update
RUN echo "===> Installing openjdk-8 for Jenkins connection and git" && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y git

# ----------------#
# Install Python  #
# ----------------#

RUN	apt-get -y update
RUN echo "===> Installing Python 3" && \
	apt-get -y install python3-pip && \
	python3 -V && \
	pip3 -V

RUN echo "===> Install pipend" && \
	pip3 install --upgrade pipenv
