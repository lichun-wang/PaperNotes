## 安装python3.6

```
# install python and pip, don't modify this, modify install_python_package.sh
apt-get update
apt-get install -y python-dev

# python 3.6
apt-get install -y software-properties-common

#add-apt-repository ppa:jonathonf/python-3.6
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python-pip python-dev python3.6 python3.6-dev

rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

## Install pip
cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py
```

