#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd
import numpy as np
import pymysql
import sys
from sqlalchemy import create_engine


# # 0、数据准备

# In[145]:


## 读取数据
datapath='D:/Personalized search engine/data'
searchlog=pd.read_csv(datapath+'/Search_history_query.csv',low_memory=False) 
user=pd.read_csv(datapath+'/user_relate_features.csv',low_memory=False)
course=pd.read_csv(datapath+'/course_features.csv',low_memory=False)
group=pd.read_csv(datapath+'/group_features.csv',low_memory=False)


# In[146]:


## 数据处理
#日志
searchlog=searchlog[searchlog['$date']=='2018-11-07']
#searchlog=searchlog[searchlog['$date']=='2018-11-07']
searchlog=searchlog[searchlog['$code'].isin(['watch_video','group_view_page','search_btn_click'])]#选出目标日志
searchlog=searchlog[searchlog['$tid'].notnull()]
#用户表
user=user.rename(columns={'resourceid':'$tid'})
tg_uid=pd.DataFrame(searchlog['$tid'].drop_duplicates())#选出目标用户

user=pd.merge(tg_uid,user,on='$tid',how='left')
user=user[user['$tid'].notnull()].set_index('$tid',drop=False)#tid设为索引同时保留字段
#课程课组表
course=course.rename(columns={'uid':'courseid'})
group=group.rename(columns={'uid':'groupid'})


# In[147]:


## 数据分段
#课程时长分段
course.loc[course['total_time']<300,'time_section']=0 
course.loc[(course['total_time']>=300)&(course['total_time']<480),'time_section']=1 
course.loc[(course['total_time']>=480)&(course['total_time']<600),'time_section']=2 
course.loc[course['total_time']>=600,'time_section']=3 
#线上线下
course['line_type']=0 #线上
course.loc[ (course['start_at'].notnull()) |(course['place'].notnull()) ,'line_type']=1 #线下
#价格分段（当前收费课组不多，暂时还没确定分段标准）
group['price_type']=0 #免费
group.loc[ group['price']!=-1,'price_type']=1 #收费


# # 1、提取用户兴趣

# In[148]:


#统一处理类别一致的字段
def consistent_Categories(table,index_group,user):
    for index_name in index_group:
        tmp=table.groupby(['$tid',index_name]).size() #分组统计
        tmp=tmp.unstack().fillna(0) #数据透视，na替换为0
        tmp[index_name]=[Counter(tmp.iloc[i]) for i in range(len(tmp))]#每行数据由series转为Counter
        user=user.join(tmp[index_name])
    return user

#统一处理类别不一致的字段，如creator_id,group_id
from collections import Counter
def unconsistent_Categories(table,index_group,user):
    for index_name in index_group: 
        grouped=table[index_name].dropna().groupby('$tid') #先删除na再分组统计，减少不必要的长度
        count=grouped.aggregate(lambda x:','.join(x)).str.split('、|,| ') #逗号连接每行，空格和顿号是每行keyword的分隔符 
        count=count.apply(lambda x:Counter(x))#每行数据为Counter类型，方便后面求和
        user=user.join(count)
    return user


# ## 1.1 从观看视频事件提取多种偏好

# In[149]:


## 提取数据，拼接课程表、课组表
watch_video=searchlog[searchlog['$code']=='watch_video'] #播放视频的用户日志
watch_video=pd.merge(watch_video[['$tid','courseid','groupid']] ,course[['courseid','video_type','line_type','time_section','is_vip','type','keyword','creator_uid','private_own','is_test_course']],on=['courseid'],how='left')
watch_video=pd.merge(watch_video,group[['groupid','price_type']],on=['groupid'],how='left')
watch_video=watch_video.set_index('$tid').rename(columns={'groupid':'groupid1','keyword':'keyword1'})


# In[150]:


## 统计分析
index_group=['video_type','line_type','time_section', 'is_vip', 'type', 'private_own', 'is_test_course','price_type'] 
user=consistent_Categories(watch_video,index_group,user)
index_group=['creator_uid','groupid1','keyword1'] 
user=unconsistent_Categories(watch_video,index_group,user)


# ## 1.2 从访问课组详情页事件提取关键词偏好

# In[151]:


## 提取数据，拼接课组表
group_view_page=searchlog[searchlog['$code']=='group_view_page'] #访问课组详情页并加载成功
group_view_page=pd.merge(group_view_page[['groupid','$tid']],group[['groupid','keyword']],on=['groupid'],how='left')
group_view_page=group_view_page.set_index('$tid').rename(columns={'groupid':'groupid2','keyword':'keyword2'})


# In[152]:


## 统计分析
index_group=['groupid2','keyword2'] 
user=unconsistent_Categories(group_view_page,index_group,user)


# ## 1.3 从点击查询按钮事件提取关键词偏好

# In[153]:


## 提取数据
search_btn_click=searchlog.loc[searchlog['$code']=='search_btn_click',['$tid','keyword']] 
search_btn_click=search_btn_click.set_index('$tid').rename(columns={'keyword':'keyword3'})


# In[154]:


## 统计分析
index_group=['keyword3'] 
user=unconsistent_Categories(search_btn_click,index_group,user)


# # 2、数据融合与汇总

# In[155]:


## 数据融合
user['groupid']=user['groupid1'].add(user['groupid2'],  fill_value=Counter())# nan加任何都为nan，需替换为Counter()
user['keyword']=user['keyword1'].add(user['keyword2'],  fill_value=Counter()).add(user['keyword3'],  fill_value=Counter())
user=user.drop(columns=['keyword1','keyword2','keyword3','groupid1','groupid2'])

## 数据处理
#nan数据
index_group=['video_type', 'line_type', 'time_section','is_vip', 'type', 'private_own',
             'is_test_course', 'price_type','creator_uid', 'groupid', 'keyword']
for index_name in index_group:
    user.loc[user[index_name].isnull(),index_name]=user.loc[user[index_name].isnull(),index_name].apply(lambda x:Counter())#由float转为Counter()

#长counter数据(取top4)
user['$creator_uid']=user['creator_uid'].apply(lambda x:x.most_common(4)) 
user['$groupid']=user['groupid'].apply(lambda x:x.most_common(4))
user['$keyword']=user['keyword'].apply(lambda x:x.most_common(4))

all_columns_name=user.columns.tolist()#获取最终user表的所有列名
columns_tuple=str(tuple(all_columns_name)).replace("'","") #用于后面拼接到sql语句中


#首次建表需要，只是为了mysql读取字典
user[all_columns_name] = user[all_columns_name].astype(str)#转为str，后续写sql需要
import pymysql
import sys
from sqlalchemy import create_engine
conn = create_engine('mysql+pymysql://huqiming:hqm2018@cvte@10.21.3.23:3306/seewo_search?charset=utf8',encoding="utf-8", echo=False)
seewoedu_user_interest_model=pd.io.sql.to_sql(user,'seewoedu_user_interest_model',con=conn,if_exists='append',index=user['$tid'])
# # 3、增量写入数据库

# ## 3.1 读出SQL
# 将数据库中已计算的兴趣表user_interest_tb读取出来

# In[156]:


conn1 = create_engine('mysql+pymysql://huqiming:hqm2018@cvte@10.21.3.23:3306/seewo_search?charset=utf8',encoding="utf-8", echo=False)
sql="select * from seewoedu_user_interest_model"
user_interest_tb=pd.read_sql(sql,conn1,index_col='$tid')
conn1.connect().close()
print(user_interest_tb.shape)


# ## 3.2 老用户累加
# 若user的$tid存在于兴趣表中，则筛选出来合并相加

# In[157]:


## 提取数据
#老用户id是user_interest_tb和user的交集
user_in_interestTB=user_interest_tb[user_interest_tb.index.isin(user.index)] #提取user_interest_tb的老用户数据
user_old=user[user.index.isin(user_in_interestTB.index)] #提取user的老用户数据

## 数据累加
index_group=['video_type','line_type', 'time_section','is_vip', 'type', 'private_own', 'is_test_course', 'price_type','creator_uid', 'groupid', 'keyword']
for index_name in index_group:
    user_in_interestTB[index_name]=user_in_interestTB[index_name].apply(lambda x:eval(x))#SQL读出字段为str，用eval重新解析为Counter
    user_old[index_name]=user_old[index_name].add(user_in_interestTB[index_name],  fill_value=Counter())#数据累加

user_old[all_columns_name] = user_old[all_columns_name].astype(str)#转为str，后续写sql需要


# ## 3.3 新用户不处理
# 若user的$tid不存在兴趣表中，则为新用户数据

# In[158]:


## 提取数据
#新用户id存在于user中但不在user_interest_tb
user_new=user[user.index.isin(user_interest_tb.index)==0] 
user_new[all_columns_name] = user_new[all_columns_name].astype(str)#转为str，后续写sql需要


# ## 3.4 更新数据库

# In[159]:


## 连接数据库
conn = pymysql.connect(host='10.21.3.23', port=3306, user='huqiming', passwd='hqm2018@cvte', db='seewo_search',charset='utf8')
cur = conn.cursor()# 使用cursor()方法获取操作游标

## 替换老用户数据
sql="replace into seewoedu_user_interest_model %s values (%s);"%(columns_tuple, ','.join(['%s'] * len(all_columns_name)))
param = user_old[all_columns_name].values.tolist()
print('替换%i条老用户数据中……'%user_old.shape[0])
try:
    cur.executemany(sql,param)
    conn.commit()
except Exception as e:
    print('Error:',e)
finally:
    print('done!')

## 插入新用户数据
sql="insert into seewoedu_user_interest_model %s values (%s);"%(columns_tuple, ','.join(['%s'] * len(all_columns_name)))
param = user_new[all_columns_name].values.tolist()
print('插入%i条新用户数据中……'%user_new.shape[0])
try:
    cur.executemany(sql,param)
    conn.commit()
except Exception as e:
    print('Error:',e)
finally:
    print('done!')

## 关闭连接
conn.close()

