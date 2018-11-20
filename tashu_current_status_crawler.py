from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import json
import datetime
import csv
import sys
import urllib.request
from bs4 import BeautifulSoup
import os
import requests
import http.client

"""
    function Name : currentStatusCrawler
    - crawl the current status(the number of rest bikes) of total tashu system
"""
def currentStatusCrawler():
    url = 'https://www.tashu.or.kr/userpage/station/mapStatus.jsp?flg=main'

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")
    driver = webdriver.Chrome("./linux_chromedriver/chromedriver", chrome_options=chrome_options)

    ## connect to tashu web server
    driver.get(url)
    stationStatus = []

    jsSourcecode = ("var jsonPath = '/mapAction.do?process=statusMapView';"+
        "$.post(jsonPath, function(data){"+
        "var jsonData = eval(\"(\"+data+\")\");"+
        "dataDiv = document.createElement('div');"+
        "dataDiv.setAttribute('id', 'dataDiv');dataDiv.innerHTML = data;"+
        "document.body.appendChild(dataDiv);"+
        "})")

    # execute js code and get current status data
    data = driver.execute_script(jsSourcecode)

    time.sleep(10)
    dataDIV = driver.find_element_by_id('dataDiv')
    dataTxt = dataDIV.text

    return dataTxt

"""
    function Name : parseData
    - parse data from tashu web and convert it to pandas.DataFrame
"""
def parseData(data, currentDateTime):
    # Convert data from tashu web to json format
    jsonData = json.loads(data)
    stationData = jsonData['markers']
    returnDF = pd.DataFrame()

    for station in stationData:
        stationNum = int(station['kiosk_no'])
        cntRackTotal = 0
        cntRentable = 0

        if stationNum > 0 and stationNum < 145:
            cntRackTotal = int(station['cntRackTotal'])
            cntRentable = int(station['cntRentable'])
            kiosk_no = int(station['kiosk_no'])

            returnDF = returnDF.append({'currentDateTime':currentDateTime,'hour':currentDateTime.hour, 'kiosk_no':stationNum, 'currentRentable':cntRentable, 'currentRackTotal':cntRackTotal},ignore_index=True)

    return returnDF


def get_time_id(datetime):
    day = datetime.day
    hour = datetime.hour
    datetime_str = str(day)+".%02d"%hour+"H"
        
    return datetime_str

def weatherDataCrawler(crnt_datetime):
    # get html
    req = urllib.request.Request("http://www.weather.go.kr/weather/observation/currentweather.jsp?stn=133")
    weather_data_page = urllib.request.urlopen(req).read()

    #parse html
    soup = BeautifulSoup(weather_data_page, 'html.parser')
    trList = soup.find_all('tr')

    crnt_feature_data = {}
    for tr in trList:
        if get_time_id(crnt_datetime) in str(tr):
            tdList = tr.find_all('td')
            crnt_feature_data['temperature'] = float(tdList[5].getText())
            crnt_feature_data['rainfall'] = tdList[8].getText()
            crnt_feature_data['humidity'] = float(tdList[10].getText())
            crnt_feature_data['windspeed'] = float(tdList[12].getText().split('\'')[1])
            #crnt_feature_data['windspeed'] = tdList[11].getText()
            break

    for key, value in crnt_feature_data.items():
        if value == '\xa0':
            crnt_feature_data[key] = 0

    return crnt_feature_data

def recordOnFile(filePath, currentDateTime, current_tashu_status):
    if os.path.isfile(filePath):
        today_history = pd.read_csv(filePath)

        prev_record = today_history.loc[today_history['hour'] == currentDateTime.hour-1,]

        if prev_record.empty:
            current_tashu_status['change_of_rentable'] = 0
        else:
            current_tashu_status['change_of_rentable'] = current_tashu_status['currentRentable'] - prev_record['currentRentable']
            print(current_tashu_status)

        today_history = today_history.append(current_tashu_status, ignore_index = True)

        today_history.to_csv(filePath)

    else:
        current_tashu_status['change_of_rentable'] = 0
        current_tashu_status.to_csv(filePath)

    return current_tashu_status

def processOnREST(serverIP, serverPort,currentDateTime, current_tashu_status):
    crnt_test_dataset = {}
    
    # call weather data from file
    current_feature_data = weatherDataCrawler(currentDateTime)

    # create current test dataset..
    for key, value in current_feature_data.items():
        crnt_test_dataset[key] = value

    crnt_test_dataset['rentMonth'] = currentDateTime.month
    crnt_test_dataset['rentHour'] = currentDateTime.hour
    crnt_test_dataset['rentWeekday'] = currentDateTime.weekday()

    change_of_rentable_dict = dict()
    change_of_rentable = current_tashu_status['change_of_rentable'].tolist()
    change_of_rentable = [0]+change_of_rentable


    crnt_test_dataset['changeOfRentable'] = change_of_rentable

    http_client = http.client.HTTPConnection(serverIP+':'+str(serverPort))
    http_client.request('POST', "/", json.dumps(crnt_test_dataset), headers = {'Content-Type':'application/json'})
    print(json.dumps(crnt_test_dataset))

def main():
    currentDateTime = datetime.datetime.now()
    currentDateTime = datetime.datetime(currentDateTime.year, currentDateTime.month, currentDateTime.day, currentDateTime.hour)

    serverIP = sys.argv[1]
    serverPort = sys.argv[2]

    crnt_tashu_status = currentStatusCrawler()
    crnt_tashu_status_df = parseData(crnt_tashu_status, currentDateTime)
    tashu_status_log_dir = "./status_data/"
    os.makedirs(tashu_status_log_dir, exist_ok=True)
    tashu_status_log_fileName = str(currentDateTime.year)+"%2d"%currentDateTime.month+"%2d"%currentDateTime.day+"_tashu_record.csv"
    tashu_status_log_path = tashu_status_log_dir+tashu_status_log_fileName
    current_tashu_status = recordOnFile(tashu_status_log_path, currentDateTime, crnt_tashu_status_df)
    processOnREST(serverIP, serverPort, currentDateTime, current_tashu_status)


if __name__ == "__main__":
    main()
