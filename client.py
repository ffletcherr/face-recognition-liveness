import argparse

import requests

# Create the parser
parser = argparse.ArgumentParser(
    description='Argument for passing an image to our flask app')

# Add the arguments
parser.add_argument('--image',
                    metavar='path',
                    type=str,
                    help='the path to the image file with jpg, jpeg or png extention.')
parser.add_argument('--host',
                    type=str,
                    default='localhost')
parser.add_argument('--port',
                    type=str,
                    default='5000')
parser.add_argument('--service',
                    type=str,
                    choices=['main', 'liveness', 'identity'],
                    help="choose between ['main', 'liveness', 'identity'] services.",
                    default='main')

args = parser.parse_args()
input_path = args.image
host = args.host
port = args.port
service = args.service

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
URL = f"http://{host}:{port}/{service}"  # main, liveness, identity


def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response.json()


response = post_image(input_path)
print("response:\n", response)
