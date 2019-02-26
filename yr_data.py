import multiprocessing as mp
from six.moves import urllib
import datetime
import re
import requests

"""
Reading from YR's public weather statistics from Trondheim's weather station 688860, Voll
"""

REGEXES = dict(
    time=re.compile(r'<th scope="row">.*<strong>(.*)</strong></th>'),
    temperature=re.compile(r'<td class="temperature .*">(.*)°C</td>'),
    rain=re.compile(r'<td>(.*) mm</td>'),
    humidity=re.compile(r'<td>(.*) %</td>')
)


def get_url(location, date):
    return f'https://www.yr.no/place/{location}/almanakk.html?dato={date}'


def get_datetime(date):
    return datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M')


def retrieve_measurements_by_date(args):
    url, date = args
    measurements, measurement = [], None

    r = requests.get(url, stream=True)

    for line in r.iter_lines():
        if line:
            line = line.decode().strip()

            time_match = REGEXES['time'].search(line)
            if time_match:
                measurement = {"temperature": []}
                when = get_datetime(f'{date}T{time_match.group(1)}')
                measurement['timestamp'] = int(when.timestamp() * 1e3)
                continue

            temperature_match = REGEXES['temperature'].search(line)
            if temperature_match:
                measurement['temperature'].append(float(temperature_match.group(1)))
                continue

            rain_match = REGEXES['rain'].search(line)
            if rain_match:
                measurement['precipitation'] = float(rain_match.group(1))
                continue

            humidity_match = REGEXES['humidity'].search(line)
            if humidity_match:
                measurement['humidity'] = int(humidity_match.group(1))
                measurement['temperature'] = dict(
                    measured=measurement['temperature'][0],
                    max=measurement['temperature'][1],
                    min=measurement['temperature'][2]
                )
                measurements.append(measurement)
                measurement = None

    return {date: measurements}


def retrieve_all_measurements(start_date, end_date, location):
    urls = []
    for year in range(start_date.year, end_date.year + 1):
        for month in range(1, end_date.month + 1):
            for day in range(1, end_date.day + 1):
                urls += [(get_url(location, f'{year}-{month}-{day}'), f'{year}-{month}-{day}')]

    print(urls)
    print("No of urls:", len(urls))
    pool = mp.Pool()
    processes = pool.map_async(retrieve_measurements_by_date, urls)
    pool.close()

    return processes.get()


def main():
    location = 'Norway/Tr%C3%B8ndelag/Trondheim/Trondheim/'
    start_date = datetime.datetime(2013, 12, 31)
    #end_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2019, 2, 25)

    all_measurements = retrieve_all_measurements(start_date, end_date, location)

    with open("YR_Dataset_Trondheim_2014_2019.json", "w") as f:
        f.write(all_measurements)

if __name__ == '__main__':
    main()
