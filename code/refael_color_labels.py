import csv
import os


def write_modified_file(color='1'):
    # From the original file, create a new file with the new labels as agreed.
    if color == '1':
        path_to_file = os.path.join(os.getcwd(), 'dataset_input')
        filename = 'Refael_Color' + color + '_12_18.csv'
        with open(os.path.join(path_to_file, filename), 'r') as read_file:
            write_file = open(os.path.join(path_to_file, 'Modified_' + filename), 'w', newline='')
            w = csv.writer(write_file)
            r = csv.reader(read_file)
            row_list = [row for row in r]
            w.writerow(row_list[0])
            for row in row_list[1:]:
                if row[4] == '3.0':
                    row[4] = '0.0'
                w.writerow(row)
    elif color == '2':
        path_to_file = os.path.join(os.getcwd(), 'dataset_input')
        filename = 'Refael_Color' + color + '_12_18.csv'
        with open(os.path.join(path_to_file, filename), 'r') as read_file:
            write_file = open(os.path.join(path_to_file, 'Modified_' + filename), 'w', newline='')
            w = csv.writer(write_file)
            r = csv.reader(read_file)
            row_list = [row for row in r]
            w.writerow(row_list[0])
            for row in row_list[1:]:
                if row[4] == '6.0':
                    continue
                elif row[4] == '1.0':
                    row[4] = '0.0'
                elif 2 <= int(row[4][0]) <= 5:
                    row[4] = '1.0'
                elif int(row[4][0]) > 6:
                    row[4] = str(float(int(row[4][0]) - 5))
                w.writerow(row)
    else:
        raise ValueError('Wrong color')


if __name__ == "__main__":
    write_modified_file('2')
