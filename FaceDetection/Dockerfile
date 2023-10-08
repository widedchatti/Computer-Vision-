FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    /opt/conda/bin/conda create -n myenv python=3.9 -y && \
    /opt/conda/bin/conda init bash && \
    echo "conda activate myenv" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]
RUN source ~/.bashrc && conda activate myenv && pip install -r requirements.txt
RUN source ~/.bashrc && conda activate myenv && pip install streamlit
COPY app.py .
EXPOSE 8501
CMD ["/opt/conda/envs/myenv/bin/streamlit", "run", "app.py"]
