# Requirements

### 1. google-cloud-storage 설치

```shell
conda activate {가상환경 이름}
pip install google-cloud-storage
```

<br/>

### 2. Google Cloud API 서비스 계정 키 생성

[이 페이지](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)의 `서비스 계정 만들기` 항목을 참고하여 JSON 키 파일을 로컬에 다운로드 받습니다.

<br/>

### 3. 환경 변수 설정

2번에서 다운받은 json 파일의 경로를 환경변수로 지정합니다.<br/>
자세한 내용은 [여기](https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable)에서 확인할 수 있습니다.

```shell
export GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
```

예시:
```shell
export GOOGLE_APPLICATION_CREDENTIALS="/home/user/Downloads/service-account-file.json"
```

<br/>

# Example

참고 링크는 [여기](https://cloud.google.com/storage/docs/samples/storage-download-file?hl=ko)<br/>
아래 코드는 txt 파일(주석 처리 하였음)과 sh 파일 모두 읽어들일 수 있는 코드입니다.

```python
from google.cloud import storage
import os

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

# bucket_name, source_blob_name, destination_file_name = "awesome-gcp-nlp", "fruit.txt", "download.txt"
bucket_name, source_blob_name, destination_file_name = "awesome-gcp-nlp", "new_test.sh", "download.sh"
download_blob(bucket_name, source_blob_name, destination_file_name)

# with open("download.txt", "r") as f:
#     print(f.read())

os.system("bash download.sh")
```
