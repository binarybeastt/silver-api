FROM python:3.11.4  
# Adjust Python version based on your requirements

WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt


EXPOSE 8000  
# Adjust port if needed

CMD ["uvicorn", "main:main", "--host", "0.0.0.0", "--port", "8000"]
