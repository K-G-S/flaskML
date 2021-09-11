import pandas as pd
import argparse
import requests
import csv
import os
import grequests
from urllib.parse import parse_qsl

size = 5

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def getReadingValues(filename, outputfile, mtype):
    excel_file = filename
    data = pd.read_excel(excel_file)
    header = ['Url', 'parameter_' + mtype, 'value_' + mtype]
    with open(os.path.join('SavedTestImages', "{}.csv".format(outputfile)), 'w',
              encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        url = 'http://127.0.0.1:5000/reading-value-display?isJson=true'
        for chunk_urls in chunks(data.url, size):
            rs_best = (grequests.post(url, data={"url": img_url, "stype": "url", "mtype": mtype}) for img_url in chunk_urls)
            response_best = grequests.map(rs_best)
            for i in response_best:
                response_best = i.json()
                print(response_best)
                try:
                    row = [dict(parse_qsl(i.request.body))['url'], response_best["parameter"], float(response_best["value"])]
                    writer.writerow(row)
                except Exception as e:
                    print(e)
                    row = [dict(parse_qsl(i.request.body))['url'], response_best["parameter"], response_best["value"]]
                    writer.writerow(row)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--infile', type=str, help='file name',
                      required=True)
  args = parser.parse_args()
  filename = args.infile
  getReadingValues(filename, "extracted_readings_from_file_best", "best")
  getReadingValues(filename, "extracted_readings_from_file_light", "light")