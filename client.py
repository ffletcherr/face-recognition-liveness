import requests
# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
URL = "http://localhost:5000/main" # main, liveness, identity
def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response.json()

response = post_image("data/test.jpg")
