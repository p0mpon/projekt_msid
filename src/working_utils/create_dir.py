import os

def create_dir(name):
    result_dir = name
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)