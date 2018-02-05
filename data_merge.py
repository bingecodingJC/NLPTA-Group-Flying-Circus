# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:14:22 2018

@author: ZHANG Xin(Serene)
"""

import pandas as pd
from datetime import datetime
from datetime import timedelta


def month_transfer(month):
    # this function is to convert the type of month
    if month == 'January':
        month_number = 1
    if month == 'February':
        month_number = 2
    if month == 'March':
        month_number = 3
    if month == 'April':
        month_number = 4
    if month == 'May':
        month_number = 5
    if month == 'June':
        month_number = 6
    if month == 'July':
        month_number = 7
    if month == 'August':
        month_number = 8
    if month == 'September':
        month_number = 9
    if month == 'October':
        month_number = 10
    if month == 'November':
        month_number = 11
    if month == 'December':
        month_number = 12
    return month_number
        
def preprocess_seeking_alpha():
    # this function is to preprocess seekingalpha.csv
    seeking_alpha = pd.read_excel(r'C:\Users\76355\Desktop\NPLTA\seekingalpha.xlsx')
    seeking_alpha.columns = ['title','time','company','label','text']
    seeking_alpha['date of call'] = ''
    seeking_alpha['Ticker'] = ''
    seeking_alpha['day0'] = ''
    seeking_alpha['day1'] = ''
    seeking_alpha['day2'] = ''
    seeking_alpha['day3'] = ''
    seeking_alpha['day4'] = ''
    seeking_alpha['day5'] = ''
    
        
    call_transcript_list_text = list(seeking_alpha['text'])
    for text_id, everytext in enumerate(call_transcript_list_text):
        # for every text, dig out the date of the earning call based on the observation that the date is listed after the first 'call' in the text
        text_split = everytext.split()
        try:
            for index, value in enumerate(text_split):
                if value == 'Call':
                    date_info = str(month_transfer(text_split[index + 1])) + ' ' + text_split[index + 2][:-1] + ' ' + text_split[index + 3]
                    break 
                
            seeking_alpha.iloc[text_id, 5] = datetime.strptime(date_info, '%m %d %Y')  # convert string date information to datetime type
        except:
            print("Cannot find date information based on the position of the word'call'", text_id)
            
    
    company_list = list(seeking_alpha['company'])
    for company_id, everycompany in enumerate(company_list):
        # for every text, dig out the ticker information based on the 'company' column data
        try:
            ticker = everycompany.split("(")[-1].split(")")[0]
            seeking_alpha.iloc[company_id,6] = ticker
        except: 
            print('Cannot find ticker information from company column', everycompany)
            
    # delete null value of 'Ticker' or 'date of call'      
    seeking_alpha = seeking_alpha[seeking_alpha['date of call'] != '']
    seeking_alpha = seeking_alpha[seeking_alpha['Ticker'] != '']
    #create index for each call, the index form is tuple(ticker, date of call)
    seeking_alpha['index_label'] = seeking_alpha[['Ticker', 'date of call']].apply(tuple, axis = 1)
    return seeking_alpha

def preprocess_stock_price():
    stock_price = pd.read_csv(r'C:\Users\76355\Desktop\NPLTA\stock_price.csv')
    stock_price = stock_price.drop(['gvkey','iid','conm'],axis = 1) #62492275 rows
    stock_price = stock_price.dropna(how = 'any')       #62474239 rows
    
    stock_price['datadate'] = pd.to_datetime(stock_price['datadate'].astype(str))
    return stock_price

def data_merge():
    stock_price = preprocess_stock_price()
    seeking_alpha = preprocess_seeking_alpha()

    seeking_alpha = seeking_alpha.sort_values(by = ['Ticker'])
    seeking_alpha_ticker_list = list(set(seeking_alpha['Ticker']))
    seeking_alpha = seeking_alpha.set_index('index_label')
    for everytic in seeking_alpha_ticker_list:
        # for every ticker, extract the related stock price information
        stock_price_everydic = stock_price[stock_price['tic'] == everytic]
        stock_price_everydic = stock_price_everydic.set_index('datadate')
        single_tic_dataframe = seeking_alpha[seeking_alpha['Ticker'] == everytic]
        date_of_call_list = list(single_tic_dataframe['date of call'])
        for eachdate in date_of_call_list:
            try:
                day0 = eachdate
                seeking_alpha.loc[(everytic,eachdate),'day0'] = stock_price_everydic.loc[day0, 'prccd']
                day1 = eachdate + timedelta(days = 1)
                seeking_alpha.loc[(everytic,eachdate),'day1'] = stock_price_everydic.loc[day1, 'prccd']
                day2 = eachdate + timedelta(days = 2)
                seeking_alpha.loc[(everytic,eachdate),'day2'] = stock_price_everydic.loc[day2, 'prccd']
                day3 = eachdate + timedelta(days = 3)
                seeking_alpha.loc[(everytic,eachdate),'day3'] = stock_price_everydic.loc[day3, 'prccd']
                day4 = eachdate + timedelta(days = 4)
                seeking_alpha.loc[(everytic,eachdate),'day4'] = stock_price_everydic.loc[day4, 'prccd']
                day5 = eachdate + timedelta(days = 5)
                seeking_alpha.loc[(everytic,eachdate),'day5'] = stock_price_everydic.loc[day5, 'prccd']
            except:
                print('the date is out of the observation')
    seeking_alpha.to_csv(r'C:\Users\76355\Desktop\NPLTA\seeking_alpha_processed.csv')
    return seeking_alpha
            
    
if __name__ == '__main__':
   
    mmm = data_merge()
    
    
    
        

    
    



