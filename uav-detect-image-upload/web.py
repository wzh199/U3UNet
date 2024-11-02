import requests
from glob import glob

url = 'http://20.196.213.10:8080/upload'

data = {"type" : "front"}

session = requests.session()

while True:
    im_list = glob('./front_img'+'/*.png')
    for i, im_path in enumerate(im_list):
        print("1")
        files = {
          "file" : open(im_path, "rb"),
        }
        print("2")
        r = session.post(url, data, files=files)
        print("3")
        print(r.text)