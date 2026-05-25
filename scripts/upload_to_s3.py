import boto3
from minio import Minio
import argparse
from pathlib import Path

class S3ModelStorage:
    def __init__(self, endpoint="localhost:9000", access_key="minioadmin", secret_key="minioadmin", bucket="models", secure=False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
            print(f"Bucket '{bucket}' created")

    def upload_model(self, local_path, object_name):
        self.client.fput_object(self.bucket, object_name, local_path)
        print(f"Uploaded {local_path} to {self.bucket}/{object_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--endpoint', type=str, default='localhost:9000')
    parser.add_argument('--access_key', type=str, default='minioadmin')
    parser.add_argument('--secret_key', type=str, default='minioadmin')
    args = parser.parse_args()

    storage = S3ModelStorage(endpoint=args.endpoint, access_key=args.access_key, secret_key=args.secret_key)
    storage.upload_model(args.model_path, args.model_name)

if __name__ == '__main__':
    main()