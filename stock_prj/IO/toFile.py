
def summary_file(data):
    file_open = 'summary.dat'
    
    with open(file_open, 'a+') as file:
        file.write(data)
