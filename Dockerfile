FROM python:3.8
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
    libsndfile1 ffmpeg libsm6 libxext6 
RUN pip3 install -r requirements.txt
EXPOSE 8506
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app/main.py"]