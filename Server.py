# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 01:27:14 2017

@author: shakt
"""

from bottle import route,run, request,error,template
import json
import base64

@route('/hello', method=['POST'])
def hello():
    print("GET")
    image = request.forms.get('image')
    name = request.forms.get('name')
    fh = open("E:\\Interesting\\Code Fun Do 2017\\inputImage.jpg", "wb")
    fh.write(base64.b64decode(image))
    fh.close()
    b="hey".encode()
    return(base64.b64encode(b))

def runServer():
    run(host='192.168.0.100', port=9999)

if __name__=="__main__":
    runServer()