FROM tensorflow/tensorflow:2.21.0

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir streamlit==1.64.0 pillow keras==3.15.1
COPY app ./app
COPY models ./models

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]