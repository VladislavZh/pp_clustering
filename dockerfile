FROM pytorch/pytorch:1.7.0-cuda10.2-cudnn8-devel


RUN apt-get update -y && apt-get upgrade -y && \
apt-get install -y gcc vim 

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
RUN pip install pandas==1.2.4 scikit-learn==0.24.1 tensorboard==2.5.0

