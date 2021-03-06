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
    os.makedirs('output', exist_ok=True)

    for file in get_files('data', ext='csv'):
        station = re.findall(r'\d+', file)[0]
        with open(file, 'r') as rf:
            reader = csv.DictReader(rf)
            with open('output/{0:03d}.csv'.format(int(station)), 'w') as wf:
                writer = csv.DictWriter(wf, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    row['changeOfRentable'] = int(row['changeOfRentable'])+50
                    row['station'] = station
                    writer.writerow(row)


if __name__ == '__main__':
    main()
