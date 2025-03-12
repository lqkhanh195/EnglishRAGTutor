FROM python:3.10-slim

ENV COHERE_API_KEY ${COHERE_API_KEY}

LABEL maintainer="khanhlq"

WORKDIR /literaturechatting/

COPY app.py DataManagement.py Retrievial.py Generation.py HistoryAdding.py requirements.txt ./

RUN apt-get update -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD streamlit run app.py --server.port 8051