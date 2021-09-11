import pandas as pd
import argparse
import requests
import csv
import os
import grequests
from urllib.parse import parse_qsl


def getReadingValues(filename):
    excel_file = filename
    data = pd.read_excel(excel_file)
    header = ['Url', 'parameter_best', 'value_best']
    with open(os.path.join('SavedTestImages', "{}.csv".format('extracted_readings_from_file_best')), 'w',
              encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        url = 'http://127.0.0.1:5000/reading-value-display?isJson=true'
        rs_best = (grequests.post(url, data={"url": img_url, "stype": "url", "mtype": "best"}) for img_url in data.url)
        response_best = grequests.map(rs_best, size=20)
        for i in response_best:
            response_best = i.json()
            try:
                row = [dict(parse_qsl(i.request.body))['url'], response_best["parameter"], float(response_best["value"])]
                writer.writerow(row)
            except Exception as e:
                print(e)
                row = [dict(parse_qsl(i.request.body))['url'], response_best["parameter"], response_best["value"]]
                writer.writerow(row)

    header = ['Url', 'parameter_light', 'value_light']
    with open(os.path.join('SavedTestImages', "{}.csv".format('extracted_readings_from_file_light')), 'w',
              encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        url = 'http://127.0.0.1:5000/reading-value-display?isJson=true'
        rs_light = (grequests.post(url, data={"url": img_url, "stype": "url", "mtype": "light"}) for img_url
                   in data.url)
        response_light = grequests.map(rs_light, size=20)
        for i in response_light:
            response_light = i.json()
            try:
                row = [dict(parse_qsl(i.request.body))['url'], response_light["parameter"],
                       float(response_light["value"])]
                writer.writerow(row)
            except Exception as e:
                print(e)
                row = [dict(parse_qsl(i.request.body))['url'], response_light["parameter"],
                       response_light["value"]]
                writer.writerow(row)

        # no = 1;
        # for img_url in data.url:
        #     print(img_url, no)
        #     img_url = str(img_url)
        #     url = 'http://127.0.0.1:5000/reading-value-display?isJson=true'
        #     headers = {'content-type': 'application/x-www-form-urlencoded'}
        #     data_best = {"url": img_url, "stype": "url", "mtype": "best"}
        #     # params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}
        #
        #     response_best = requests.post(url, data=data_best, headers=headers)
        #     response_best = response_best.json()
        #
        #     data_light = {"url": img_url, "stype": "url", "mtype": "light"}
        #     # params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}
        #
        #     response_light = requests.post(url, data=data_light, headers=headers)
        #     response_light = response_light.json()
        #     try:
        #         row = [img_url, response_best["parameter"], float(response_best["value"]),
        #                response_light["parameter"], float(response_light["value"])]
        #         writer.writerow(row)
        #     except Exception as e:
        #         print(e)
        #         row = [img_url, response_best["parameter"], response_best["value"],
        #                response_light["parameter"], response_light["value"]]
        #         writer.writerow(row)
        #     no += 1
        # no = 1;





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--infile', type=str, help='file name',
                      required=True)
  args = parser.parse_args()
  print(args)
  filename = args.infile
  print(filename)
  getReadingValues(filename)