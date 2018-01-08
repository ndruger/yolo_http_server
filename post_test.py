import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--file_path',
    help='file path to send',
    required=True)
parser.add_argument(
    '-p',
    '--port',
    type=int,
    help='http port, default 8000',
    default=8000)

args = parser.parse_args()
data = open(args.file_path, 'rb').read()
res = requests.post(url='http://127.0.0.1:{}/'.format(args.port),
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})

print(res.content)
