FROM frolvlad/alpine-miniconda2

RUN apk add --no-cache build-base imagemagick
RUN apk add --update bash git curl unzip && rm -rf /var/cache/apk/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir Cython==0.24.1 mock==1.3.0 numpy==1.11.1
RUN conda install matplotlib==1.5.1 \
    && conda install tifffile -c conda-forge \
    && pip install --no-cache-dir pandas==0.18.1 pymorph==0.96 PyWavelets==0.4.0 \
    scipy==0.19.0 simplejson==3.8.2 scikit_image==0.13.0 Pillow==3.3.1 \
    SimpleITK==0.10.0 centrosome==1.0.5 ipywidgets==5.2.2 joblib==0.10.2 \
    pypng==0.0.18 mahotas==1.4.1  opencv-python==3.2.0.7 \
    git+https://github.com/jfrelinger/cython-munkres-wrapper \
    jupyter
RUN pip install numba notebook==5.4.1



EXPOSE 8888
WORKDIR /home

RUN jupyter notebook --generate-config --allow-root \
    && echo 'c.NotebookApp.allow_root = True' >> /root/.jupyter/jupyter_notebook_config.py \
    && echo 'c.NotebookApp.ip = "*"' >> /root/.jupyter/jupyter_notebook_config.py \
    && echo 'c.NotebookApp.port = 8888' >> /root/.jupyter/jupyter_notebook_config.py


CMD ["/bin/bash"]
