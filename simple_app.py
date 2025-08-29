#!/usr/bin/env python3
import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def health_check():
    return {'status': 'ok', 'message': 'BiRefNet service is running'}

@app.route('/test')
def test():
    return {'message': 'Test endpoint working'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)