from flask import Flask, jsonify, render_template, request
app = Flask(__name__)
from pyside import *

@app.route('/bridge')
def bridge():
	good=integrate()
	file1=open('test 1.txt','w')
	file1.write(str(good))
	file1.close()
	if(good==1):
		return jsonify(status="Good Squat")
	if(good==0):
		return jsonify(status="Bad Squat")

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)