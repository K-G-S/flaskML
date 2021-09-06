import pandas as pd
import argparse
import requests
import csv
import os

def getReadingValues(filename):
    excel_file = filename
    data = pd.read_excel(excel_file)
    image_urls_list = []
    parameter_best = []
    value_best = []
    value_light = []
    parameter_light = []
    for img_url in data.url:
        print(img_url)
        img_url = str(img_url)
        url = 'http://127.0.0.1:5000/reading-value-display?isJson=true'
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        data_best = {"url": img_url, "stype": "url", "mtype": "best"}
        # params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}

        response_best = requests.post(url, data=data_best, headers=headers)
        response_best = response_best.json()

        data_light = {"url": img_url, "stype": "url", "mtype": "light"}
        # params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}

        response_light = requests.post(url, data=data_light, headers=headers)
        response_light = response_light.json()
        print(response_best, response_light)
        image_urls_list.append(img_url)
        parameter_best.append(response_best["parameter"])
        value_best.append(float(response_best["value"]) if response_best["value"] and response_best["value"] != '.' else response_best["value"])
        parameter_light.append(response_light["parameter"])
        value_light.append(float(response_light["value"]) if response_light["value"] and response_light["value"] != '.' else response_light["value"])

    df = pd.DataFrame({'Url': image_urls_list, 'parameter_best': parameter_best, 'value_best': value_best,
                       'parameter_light': parameter_light, 'value_light': value_light})
    df.to_excel(os.path.join('SavedTestImages', "{}.xlsx".format('extracted_readings_from_file')), sheet_name='sheet1', index=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--infile', type=str, help='file name',
                      required=True)
  args = parser.parse_args()
  print(args)
  filename = args.infile
  print(filename)
  getReadingValues(filename)