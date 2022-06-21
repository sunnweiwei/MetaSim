import os
import time

num_process = 54

while True:
    time.sleep(0.1)
    for idx in range(num_process):
        path = f'system_redial/{idx}/'
        input_path = path + 'input/'
        output_path = path + 'output/'
        file_list = os.listdir(input_path)
        for file in file_list:
            timestap = int(file.split('.')[0])
            time_now = int(time.time())
            if time_now - timestap > 10:
                try:
                    os.remove(input_path + file)
                except:
                    continue
        file_list = os.listdir(output_path)
        for file in file_list:
            timestap = int(file.split('.')[0])
            time_now = int(time.time())
            if time_now - timestap > 10:
                try:
                    os.remove(output_path + file)
                except:
                    continue