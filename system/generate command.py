command = ''
for idx in range(54):
    command += f'nohup python gpt_server.py --process_id {idx} >> id_{idx} 2>&1 &\n'
print(command)