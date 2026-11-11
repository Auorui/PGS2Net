from pyzjr.data import move_files, copy_files, find_files


target_dir = r'E:\PythonProject\TransmissionLine\data\Output\Image'
save_dir = r'E:\PythonProject\TransmissionLine\data\Output\Images'
file_list = find_files(target_dir, '.png')

move_files(file_list, save_dir)

