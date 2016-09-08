# gtx1080_tensorflow
파스칼 아키텍쳐 GPU (GTX 1060, 1070, 1080 시리즈)에서 Tensorflow 설치하기 

## 겪었던 문제들
#### 1. Ubuntu 16.04에서 드라이버문제로 부팅시 검은 화면만 출력되는 현상
메인보드에서 그래픽카드를 분리한다음 부팅하고 nvidia-367드라이버를 설치한다.

        $ sudo add-apt-repository ppa:graphics-drivers/ppa
        $ sudo apt-get update
        $ sudo apt-get install nvidia-367

종료후 그래픽카드를 연결하고 부팅하면 정상적으로 우분투가 열린다.

#### 2. GTX1080 + CUDA7.5 + Cudnn 4.x 설치시 연산 오류
제목과 같은 설정으로 [이 링크](http://tech.marksblogg.com/tensorflow-nvidia-gtx-1080.html) 를 참고하여 
성공적으로 설치했으나, [CNN 예제](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html)를 실행할경우 accuracy가 낮게 나온다.
관련 정보를 검색해보니 파스칼 계열 그래픽카드 에서 CUDA 8.0RC + Cudnn 5.x로 설치해야 정상 동작한다고한다.

---------------------------------------

## GTX 1080 + Tensorflow v0.10 + Cuda8 + Cudnn5.1 설치 
#### 설치환경 
| 구분      | 사양  |
| --------- | -----------------------|
| CPU | i7-6700 |
| GPU | GTX-1080 |
| RAM | 16GB * 2 |
| OS | Ubuntu 16.04 |

#### 드라이버 설치
우분투 부팅후 검은화면만 출력될경우 그래픽카드를 분리하고 내장그래픽으로 부팅한다.

        $ sudo add-apt-repository ppa:graphics-drivers/ppa
        $ sudo apt-get update
        $ sudo apt-get install nvidia-367

그래픽카드를 연결하고 재부팅한다.

#### CUDA 설치
[CUDA 웹사이트](developer.nvidia.com/cuda-release-candidate-download) 에서 로그인 후 Linux > x86_64 > Ubuntu > 16.04 > runfile(local) 에서 CUDA 8.0과 Patch1을 받는다.

        $ sudo sh cuda_8.0.27_linux.run --override
        
        Do you accept the previously read EULA?
        accept/decline/quit: accept
        Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 361.77?
        (y)es/(n)o/(q)uit: n
        Install the CUDA 8.0 Toolkit?
        (y)es/(n)o/(q)uit: y
        Enter Toolkit Location
        [ default is /usr/local/cuda-8.0]: enter
        Do you want to install a symbolic link at /usr/local/cuda?
        (y)es/(n)o/(q)uit:y
        Install the CUDA 8.0 Samples?
        (y)es/(n)o/(q)uit:y
        Enter CUDA Samples Location
        [ defualt is /root ]: enter
        
        $ sudo sh cuda_8.0.27.1_linux.run
        
        Do you accept the previously read EULA?
        accept/decline/quit:accept
        Enter CUDA Toolkit installation directory
        [ default is /usr/local/cuda-8.0 ]: enter
        

#### 경로 설정
CUDA 경로를 등록해준다.

        $ sudo gedit /home/유저이름/.bashrc

가장 아래에 다음과같은 명령을 추가한다.

        export CUDA_HOME=/usr/local/cuda-8.0
        export PATH=/usr/local/cuda-8.0/bin${PATH:+:${path}}
        export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

bashrc를 다시 불러와 경로가 제대로 등록되있나 확인해본다.

        $ sudo source ~/.bashrc
        $ sudo echo $CUDA_HOME
        /usr/local/cuda-8.0
  
  CUDA 설치 확인

        $ sudo nvidia-smi

현재 GPU 정보 등이 출력된다.


#### CUDNN 설치

[CUDA 웹사이트](developer.nvidia.com/cudnn) 에서 로그인 후 cudnn 5.1 버전을 다운받는다.

        $ sudo tar xvzf cudnn-8.0-linux-x86-v5.1.tgz
        $ cd cuda
        $ sudo cp include/cudnn.h /usr/local/cuda-8.0/include/
        $ sudo cp lib64/libcudnn* /usr/local/cuda/lib64/

#### Tensorflow v0.10 설치

python 기본 환경과 git을 설치한다.

        $ sudo apt-get install python-dev python-pip python-numpy swig python-wheel git

bazel과 java 설치

        $ sudo add-apt-repository ppa:webupd8team/java
        $ sudo apt-get update
        $ sudo apt-get install oracle-java8-installer
        
        $ sudo echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
        $ sudo curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
        $ sudo apt-get update && sudo apt-get install bazel
        $ sudo apt-get upgrade bazel


tensorflow v0.10 다운로드

        $ sudo git clone -b r0.10 https://github.com/tensorflow/tensorflow
        $ cd tensorflow

CROSSTOOL 파일 수정
third_party/gpus/crosstool/CROSSTOOL 파일을 열어 
cxx_builtin_include_directory가 있는 라인을 검색후 아래와 같이 추가한다.

        cxx_builtin_include_directory: "/usr/lib/gcc/"
        cxx_builtin_include_directory: "/usr/local/include"
        cxx_builtin_include_directory: "/usr/include"
        cxx_builtin_include_directory: "/usr/local/cuda-8.0/include"
        tool_path { name: "gcov" path: "/usr/bin/gcov" }

configure 스크립트를 실행한다. GTX 10XX 계열은 compute capability가 6.1이다.

        $ sudo ./configure
        
        Do you wish to build TensorFlow with Google Cloud Platform support? [y/n] N
        Do you wish to build TensorFlow with GPU suppport? [y/n] y
        Please specify with gcc should be used by nvcc as the host compiler.
        [Default is /usr/bin/gcc]: enter
        Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system Default]: enter
        Please specify the location where CUDA toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: enter
        Please specify the Cudnn version you wnat to use. [Leave empty to use system default]:enter
        Please specify the location where cuDNN library is installed. Refer to README.md for more details. [Default is /usr/local/cuda] : enter
        Please note that each additional compute capability significantly increases your build time and binary size
        [Default is : "3.5,5.2"] 6.1

bazel을 이용해 tensorflow 를 빌드시킨다.

        $ sudo bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
        $ sudo bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
        $ sudo pip install /tmp/tensorflow_pkg/tensorflow-0.10.0-py2-none-any.whl

성공적으로 설치됬다면 테스트를 해보자

        $ python
        > import tensorflow as tf
        > hello = tf.constant('Hello, world!')
        > sess = tf.Session()
        > print(sess.run(hello))
        Hello, world!

## import error : pywrap_tensorflow
만약 pywrap_tensorflow.py를 찾을수 없다고 나온다면 

bashrc에 경로 설정을 잘못했을 가능성이 농후하다.

경로 설정을 다시하고 다시 빌드 해보자.
        
## import error : libcudart.so.8.0

        $ source ~/.bashrc
        $ python
        > import tensorflow 
        successfully opend CUDA library libcublas.so locally
        

## 참고한 링크
- http://tech.marksblogg.com/tensorflow-nvidia-gtx-1080.html
- https://marcnu.github.io/2016-08-17/Tensorflow-v0.10-installed-from-scratch-Ubuntu-16.04-CUDA8.0RC-cuDNN5.1-1080GTX/
- http://textminingonline.com/dive-into-tensorflow-part-iii-gtx-1080-ubuntu16-04-cuda8-0-cudnn5-0-tensorflow
- https://github.com/est-ai/tensorflow-on-pascal
- http://stackoverflow.com/questions/38036837/extremly-low-accuracy-in-deep-mnist-for-experts-using-pascal-gpu




