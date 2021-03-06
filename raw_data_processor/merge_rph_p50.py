import os
import csv
import re


def get_files(path, ext='', recursive=False):
    path_list = [path]

    while len(path_list) > 0:
        cpath = path_list.pop()
        with os.scandir(cpath) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_file():
                    if entry.name.endswith(ext):
                        yield entry.path
                    else:
                        if recursive:
                            path_list.append(entry.path)


def main():
    fieldnames = ['station', 'rentMonth', 'rentHour', 'rentWeekday', 'temperature',
                  'humidity', 'windspeed', 'rainfall', 'changeOfRentable']
    min_r = 100
    max_r = -100
    with open('output.csv', 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        for file in get_files('data', ext='csv'):
            station = re.findall(r'\d+', file)[0]
            with open(file, 'r') as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    rph = int(row['changeOfRentable'])+50
                    if min_r > rph:
                        min_r = rph
                    if max_r < rph:
                        max_r = rph
                    row['changeOfRentable'] = rph
                    row['station'] = station
                    writer.writerow(row)
    print(min_r, max_r)


if __name__ == '__main__':
    main()
