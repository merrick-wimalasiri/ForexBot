#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fxcmpy
fxcmpy.__version__
import configparser
import datetime 


# In[2]:


TOKEN = '434e3e87b7324219bedc0aed785c3dd24c6a0578'
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')


# In[3]:


print(con.get_instruments())


# In[4]:


data = con.get_candles('EUR/USD', period='m1', number=250)
data.head()
parse_dates("=['data'],index_col='data')")


# In[5]:


data.tail()


# In[6]:


from pylab import plt
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


data['askclose'].plot(figsize=(20, 6));


# In[8]:


con.subscribe_market_data('EUR/USD')


# In[49]:


con.get_prices('EUR/USD')


# In[9]:


candles=con.get_candles(offer_id=1, period='m1', number=50)
candles


# In[51]:


candles[['askclose']].plot(figsize=(10,6));


# In[52]:


data=candles[['askopen','askhigh','asklow','askclose']]
data.columns=['open','high','low','close']
data.info()


# In[10]:


import cufflinks as cf


# In[11]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[12]:


qf=cf.QuantFig(data, title='EUR/USD', legend='top',
              name='EUR/USD', datalegend=True)


# In[13]:


iplot(qf.iplot(asFigure=True))


# In[14]:


qf.add_bollinger_bands(periods=10,boll_std=2, colors=['magenta', 'grey'], fill=True)
qf.data.update()


# In[15]:


iplot(qf.iplot(asFigure=True))


# In[16]:


import numpy as np
import pandas as pd
import pytz


# In[19]:


candles=con.get_candles(offer_id=1, period='m1', number=400)
candles.tail(10)


# In[20]:


data=pd.DataFrame(candles[['askclose', 'bidclose']].mean(axis=1),columns=['midclose'])


# In[21]:


iplot(data.iplot(asFigure=True))


# In[46]:


data['returns']=np.log(data/data.shift(1))


# In[23]:


lags = 6
cols = []
for lag in range (1, lags + 1):
    col='lag_%s'%lag
    data[col]=data['returns'].shift(lag)
    cols.append(col)


# In[24]:


data.head()


# In[25]:


data.tail()


# In[26]:



data['direction']=np.sign(data['returns'])
to_plot=['midclose', 'returns', 'direction']
data[to_plot].iloc[:100].plot(figsize=(10,6),
        subplots=True,style=['-','-','ro'], title='EUR/USD');


# In[27]:


np.digitize(data[cols],bins=[0])[:400]


# In[69]:


2 ** 6


# In[70]:


data.dropna(inplace=True)


# In[71]:


data.dropna(inplace=True)


# In[28]:


from sklearn import svm


# In[42]:


model=svm.SVC(C=100)


# In[43]:


get_ipython().run_line_magic('time', "model.fit(np.sign(data[cols]), np.sign(data['returns']))")


# In[44]:


data.info()


# In[45]:


pred=model.predict(np.sign(data[cols]))
pred[:400]


# In[38]:


model.classes_


# In[78]:


pred_proba[:10]


# In[79]:


data['position'] = pred


# In[80]:


data['strategy']=data['position']*data['returns']


# In[81]:


data[['returns','strategy']].cumsum().apply(np.exp).plot(figsize=(10,6));


# In[82]:


data['position'].value_counts()


# In[83]:


data['position'].diff().value_counts()


# In[84]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[85]:


mu=data['returns'].mean()
v=data['returns'].std()
bins=[mu-v,mu,mu+v]
train_x,test_x,train_y,test_y=train_test_split(
    data[cols].apply(lambda x: np.digitize(x, bins=bins)),
    np.sign(data['returns']),
    test_size=0.5,random_state=111)


# In[86]:


train_x.sort_index(inplace=True)
train_y.sort_index(inplace=True)
test_x.sort_index(inplace=True)
test_y.sort_index(inplace=True)


# In[87]:


train_x.head()


# 

# In[88]:


model.fit(train_x, train_y)


# In[89]:


train_pred=model.predict(train_x)


# In[90]:


accuracy_score(train_y,train_pred)


# In[91]:


test_pred=model.predict(test_x)


# In[92]:


accuracy_score(test_y,test_pred)


# In[93]:


data.loc[test_x.index][['returns','strategy']].cumsum().apply(np.exp).plot(figsize=(10,6));


# In[94]:


def output(data, dataframe):
    print('%3d | new values received for %s | %s, %s, %s, %s, %s'
          % (len(dataframe), data['Symbol'], pd.to_datetime(int(data['Updated']), unit='ms'),
             data['Rates'][0],data['Rates'][1],data['Rates'][2],data['Rates'][3]))


# In[95]:


con.subscribe_market_data('EUR/USD', (output,))


# In[96]:


con.unsubscribe_market_data('EUR/USD')


# In[97]:


con.get_open_positions()


# In[98]:


order=con.create_market_buy_order('EUR/USD',100)


# In[99]:


order.get_currency()


# In[100]:


order.get_isBuy()


# In[101]:


cols=['tradeId', 'amountK', 'currency', 'grossPL', 'isBuy']


# In[102]:


con.get_open_positions()[cols]


# In[103]:


order=con.create_market_buy_order('USD/JPY',50)


# In[104]:


con.get_open_positions()[cols]


# In[105]:


order=con.create_market_sell_order('EUR/USD',25)


# In[106]:


order=con.create_market_buy_order('USD/JPY',50)


# In[107]:


con.get_open_positions()[cols]


# In[108]:


order=con.create_market_sell_order('EUR/USD',50)


# In[109]:


con.get_open_positions()[cols]


# In[110]:


con.close_all_for_symbol ('USD/JPY')


# In[111]:


con.get_open_positions()[cols]


# In[112]:


con.close_all()


# In[113]:


con.get_open_positions()


# In[114]:


lags=3


# In[115]:


def generate_features(df, lags):
    df['Returns'] = np.log(df['Mid'] / df['Mid'].shift(1))
    cols = []
    for lag in range(1, lags+1):
        col='lag_%s' % lag
        df[col]=np.sign(df['Returns'].shift(lag))
        cols.append(col)
        df.dropna(inplace=True)
        return df, cols


# In[116]:


candles=con.get_candles('EUR/USD',period='m1',number=1000)


# In[117]:


data=pd.DataFrame(candles[['askclose','bidclose']].mean(axis=1),
                 columns=['Mid'])


# In[118]:


data, cols = generate_features(data, lags)


# In[119]:


data.tail()


# In[120]:


model.fit(data[cols],np.sign(data['Returns']))


# In[121]:


model.predict(data[cols])[:10]


# In[122]:


data[cols].iloc[-1].values


# In[123]:


model.predict(data[cols].iloc[-1].values.reshape(1,-1))


# In[124]:


TOKEN = '434e3e87b7324219bedc0aed785c3dd24c6a0578'
con2= fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')


# In[125]:


to_show=['tradeID','amountK','currency','grossPL','isBuy']


# In[126]:


ticks=0
position=0
tick_data=pd.DataFrame()
tick_resam=pd.DataFrame()


# In[127]:


def automated_trading(data,df):
    global lags,model,ticks
    global tick_data, tick_resam, to_show
    global position
    ticks +=1
    t=datetime.datetime.now()
    if ticks % 5==0:
        print('%3d | %s | %7.5f | %7.5f' % (ticks, str(t.time()),
                                   data['Rates'][0],data['Rates'][1]))
    #collecting tick data
    tick_data=tick_data.append(pd.DataFrame(
            {'Bid': data['Rates'][0],'Ask':data['Rates'][1],
            'High': data['Rates'][2],'Low':data['Rates'][3]},
            index=[t]))
    #Resampling Tick Data
    tick_resam=tick_data[['Bid','Ask']].resample('5s',label='right').last().ffill()
    tick_resam['Mid']=tick_resam.mean(axis=1)
    if len(tick_resam)>lags+2:
        #Generating Signal
        tick_resam, cols=generate_features(tick_resam,lags)
        tick_resam['Prediction']=model.predict(tick_resam[cols])
        #Entering a Long position
        if tick_resam['Prediction'].iloc[-2]>=0 and position == 0:
            print('going long(for the first time)')
            position = 300
            order=con2.create_market_buy_order('EUR/USD', 25)
            trade=True
        elif tick_resam['Prediction'].iloc[-2]>=0 and position == -1:
            print('going long')
            position = 300
            order=con2.create_market_buy_order('EUR/USD',50)
            trade=True
        #Entering a short position
        elif tick_resam['Prediction'].iloc[-2] <= 0 and position ==0:
            print('going short')
            position=-300
            order=con2.create_market_sell_order('EUR/USD',25)
        elif tick_resam['Prediction'].iloc[-2] <= 0 and position == 1:
            print('going short')
            position = -300
            order=con2.create_market_sell_order('EUR/USD',50)
        if ticks > 150:
            con.unsubscribe_market_data('EUR/USD')
            print('closing out all positions')
            try:
                con2.close_all()
            except:
                pass
            
            
        


# In[135]:


con.subscribe_market_data('EUR/USD',(automated_trading,))


# In[134]:


try:
    print(con2.get_open_positions()[to_show])
except:
    print('no open position')


# In[132]:


con.get_open_positions().T


# In[ ]:




