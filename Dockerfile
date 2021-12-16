FROM python:3.8-buster

RUN mkdir /opt/streamlit/
WORKDIR /opt/streamlit/
ADD requirements.txt ./
ADD src ./src
ADD .streamlit ./.streamlit
RUN pip install --upgrade pip
# Warning: streamlit neet to bokeh == 2.2
# https://github.com/streamlit/streamlit/issues/2156
RUN pip install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/opt/streamlit/

CMD streamlit run src/app.py
