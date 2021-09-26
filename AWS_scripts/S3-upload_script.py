from user_Key import access_key,secret_access_key

import boto3
import os


client = boto3.client('s3', aws_access_key_id = access_key,
                        aws_secret_access_key = secret_access_key)


for file in os.listdir():
    if '.py' in file:
        upload_file_bucket = 'trialandtest'
        upload_file_key = 'python scripts/' + str(file)
        client.upload_file(file, upload_file_bucket, upload_file_key)
    elif '.csv' in file:
        upload_file_bucket_csv = 'time1231'
        upload_file_key_csv = ' csv_files/' + str(file)
        client.upload_file(file, upload_file_bucket_csv, upload_file_key_csv)
    # elif 'file type' in file:
    #     upload_file_bucket_csv = 'bucketname'
    #     upload_file_key_csv = ' foldername' + str(file)
    #     client.upload_file(file, upload_file_bucket_csv, upload_file_key_csv)





