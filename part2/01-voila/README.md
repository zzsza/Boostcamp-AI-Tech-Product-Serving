# Voila
[Voila](https://voila.readthedocs.io/en/stable/)를 이용한 간단한 프로토타이핑에 대해 학습합니다


## Installation
- pip를 활용한 설치
    ```shell
    pip3 install voila
    ```

- JupyterLab 사용한다면
    ```shell
    jupyter labextension install @jupyter-voila/jupyterlab-preview 
    ```

- jupyter Notebook이나 Jupyter Server를 사용한다면
    ```
    jupyter serverextension enable voila --sys-prefix
    ```

- nbextension도 사용 가능하도록 하고 싶다면 다음과 같이 설정
    ```
    voila --enable_nbextensions=True
    jupyter notebook --VoilaConfiguration.enable_nbextensions=True
    ```

## References
- [voila-gallery](https://voila-gallery.org/): Voila를 활용한 다양한 예제를 확인해보세요
