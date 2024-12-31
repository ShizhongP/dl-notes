# Huggingface 下载数据或模型教程

现在大部分的开源模型和数据基本上在huggingface上都能找到，手动下载模型通常比较繁琐，下面介绍两种方式下载模型或者代码

## 方式一: snapshot_download 代码

代码

```python
from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="bigscience/bloom-560m",
  local_dir="/data/user/test",
  local_dir_use_symlinks=False,
  proxies={"https": "http://localhost:7890"}
)
```

1. repo_id 为模型名称，上面的`bigscience/bllom-560m`的意思是`bigscience`用户下的`bllom-560m`模型
2. local_dir 为模型要下载的本地路径（可选，但是建议自己选个下载路径）
3. local_dir_use_symlinks 意味当本地下载路径已经存在该模型时是否创建符号连接（可选）
4. proxies 使用你的代理服务，如果有的话，如果不使用代理可以不加该参数（可选）

## 方式二: huggingface-cli  命令行（推荐）

如果按照下面方式下载过程中出现`Connect error`那么可能是主机和huggingface的的网络连接问题，因为huggingface在外网，通常需要设置代理或者配置huggingface国内镜像（教程见下文）

1. 安装huggingface-cli, transformer

   ```shell
   pip install transformer huggingface-cli
   ```

2. huggingface-cli 下载数据，token获取方式见文末

   ```shell
   huggingface-cli download --repo-type dataset --token 你的token --resume-download 数据或者模型名称 --cache-dir 本地下载路径 
   ```

3. huggingface-cli 下载模型

   ```shell
   huggingface-cli download --repo-type model --token 你的token --resume-download 模型名称 --cache-dir 本地下载路径 --local-dir-use-symlinks False
   ```

## huggingface token获取

登陆huggingface ,后点击个人头像，然后点击`Access token`

<img src="/home/psz/workspace/AI/My-project/dl-notes/notes/assets/image-20241230205657898.png" alt="image-20241230205657898" style="zoom:50%;" />

然后点击create new token创建一个token

Token name 为这个token设置一个名称，不知道选择，按照下面的勾选可以了, 然后滑到最底下Create token即完成创建。**记住创建的时候要复制并记录好，之后这个token不可以被再次查看**

<img src="/home/psz/workspace/AI/My-project/dl-notes/notes/assets/image-20241230205903864.png" alt="image-20241230205903864" style="zoom:50%;" />

## 配置huggingface 国内镜像（可选）

参考:https://padeoe.com/huggingface-large-models-downloader/

因为huggingface服务器在外网，所以一般访问会比较慢（或者不可访问），经常需要开代理，因此可以使用huggingface的国内镜像网站，通过设置环境变量可以切换默认访问的服务器

1. 配置环境变量

   ```shell
   export HF_ENDPOINT="https://hf-mirror.com"
   ```

2. 如果需要提高下载速度，可以配置hf_transfer(可选)

   ```shell
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```

配置好镜像网站之后，就会自动到镜像网站去下载数据或模型了

## 配置主机的代理服务

根据自己主机配置的代理服务替换`proxy_address`和`port`

```shell
export HTTP_PROXY="http://proxy_address:port"
export HTTPS_PROXY="http://proxy_address:port"
```

