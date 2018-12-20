#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
import numpy as np


# # 0、数据准备

# In[94]:


## 读取数据
datapath='D:/Personalized search engine/data'
searchlog=pd.read_csv(datapath+'/Search_history_query.csv',low_memory=False) 
user=pd.read_csv(datapath+'/user_relate_features.csv',low_memory=False)
course=pd.read_csv(datapath+'/course_features.csv',low_memory=False)
group=pd.read_csv(datapath+'/group_features.csv',low_memory=False)


# In[95]:


## 数据处理
#日志
searchlog=searchlog[searchlog['$date']=='2018-11-06']
searchlog=searchlog[searchlog['$code'].isin(['watch_video','group_view_page','search_btn_click'])]#选出目标日志
#用户表
user=user.rename(columns={'resourceid':'$tid'})
tg_uid=pd.DataFrame(searchlog['$tid'].drop_duplicates())#选出目标用户
user=pd.merge(tg_uid,user,on='$tid',how='left')
user=user[user['$tid'].notnull()].set_index('$tid')
#课程课组表
course=course.rename(columns={'uid':'courseid'})
group=group.rename(columns={'uid':'groupid'})


# In[96]:


## 数据分段
#课程时长分段
course.loc[course['total_time']<300,'time_section']=0 #
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

# In[97]:


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

# In[98]:


## 提取数据，拼接课程表、课组表
watch_video=searchlog[searchlog['$code']=='watch_video'] #播放视频的用户日志
watch_video=pd.merge(watch_video[['$tid','courseid','groupid']] ,course[['courseid','video_type','line_type','time_section','is_vip','type','keyword','creator_uid','private_own','is_test_course']],on=['courseid'],how='left')
watch_video=pd.merge(watch_video,group[['groupid','price_type']],on=['groupid'],how='left')
watch_video=watch_video.set_index('$tid').rename(columns={'groupid':'groupid1','keyword':'keyword1'})


# In[99]:


## 统计分析
index_group=['video_type','line_type','time_section', 'is_vip', 'type', 'private_own', 'is_test_course','price_type'] 
user=consistent_Categories(watch_video,index_group,user)
index_group=['creator_uid','groupid1','keyword1'] 
user=unconsistent_Categories(watch_video,index_group,user)


# ## 1.2 从访问课组详情页事件提取关键词偏好

# In[100]:


## 提取数据，拼接课组表
group_view_page=searchlog[searchlog['$code']=='group_view_page'] #访问课组详情页并加载成功
group_view_page=pd.merge(group_view_page[['groupid','$tid']],group[['groupid','keyword']],on=['groupid'],how='left')
group_view_page=group_view_page.set_index('$tid').rename(columns={'groupid':'groupid2','keyword':'keyword2'})


# In[101]:


## 统计分析
index_group=['groupid2','keyword2'] 
user=unconsistent_Categories(group_view_page,index_group,user)


# ## 1.3 从点击查询按钮事件提取关键词偏好

# In[102]:


## 提取数据
search_btn_click=searchlog.loc[searchlog['$code']=='search_btn_click',['$tid','keyword']] 
search_btn_click=search_btn_click.set_index('$tid').rename(columns={'keyword':'keyword3'})


# In[103]:


## 统计分析
index_group=['keyword3'] 
user=unconsistent_Categories(search_btn_click,index_group,user)


# # 2、数据融合与汇总

# In[104]:


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
user['$keyword']=user['keyword'].apply(lambda x:x.most_common(4))
user['$groupid']=user['groupid'].apply(lambda x:x.most_common(4))
user['$creator_uid']=user['creator_uid'].apply(lambda x:x.most_common(4)) 


# # 3、增量写入数据库

# ## 3.1 读出SQL
# 将数据库中已计算的兴趣表user_interest_tb读取出来

# In[105]:


import pymysql
import sys
from sqlalchemy import create_engine
conn = create_engine('mysql+pymysql://huqiming:hqm2018@cvte@10.21.3.23:3306/seewo_search?charset=utf8',encoding="utf-8", echo=False)

sql="select * from seewoedu_user_interest_model"
user_interest_tb=pd.read_sql(sql,conn,index_col='$tid')


# ## 3.2 老用户累加
# 若user的$tid存在于兴趣表中，则筛选出来合并相加，替换user_interest_tb的旧数据

# In[106]:


## 提取数据
#老用户id是user_interest_tb和user的交集
user_in_interestTB=user_interest_tb[user_interest_tb.index.isin(user.index)] #提取user_interest_tb的老用户数据
user_old=user[user.index.isin(user_in_interestTB.index)] #提取user的老用户数据

## 数据累加
index_group=['video_type','line_type', 'time_section','is_vip', 'type', 'private_own', 'is_test_course', 'price_type','creator_uid', 'groupid', 'keyword']
for index_name in index_group:
    user_in_interestTB[index_name]=user_in_interestTB[index_name].apply(lambda x:eval(x))#SQL读出字段为str，用eval重新解析为Counter
    user_in_interestTB[index_name]=user_in_interestTB[index_name].add(user_old[index_name],  fill_value=Counter())#数据累加

## 数据刷新
user_in_interestTB = user_in_interestTB.astype(str)#转为str，后续写sql需要
user_interest_tb[user_interest_tb.index.isin(user_in_interestTB.index)]=user_in_interestTB #直接在user_interest_tb里修改


# ## 3.3 新用户拼接
# 若user的$tid不存在兴趣表中，则直接append到user_interest_tb中

# In[107]:


## 提取数据
#新用户id存在于user中但不在user_interest_tb
user_new=user[user.index.isin(user_interest_tb.index)==0] 

## 刷新数据
index_group=user_new.columns.values.tolist() #全部字段
user_new[index_group] = user_new[index_group].astype(str)#转为str，后续写sql需要
user_interest_tb=user_interest_tb.append(user_new,sort=False)#直接拼接到user_interest_tb


# ## 3.4 写入SQL
# 将user_interest_tb以replace的形式重新写入到数据库中

# In[108]:


user_interest_tb['$tid']=user_interest_tb.index
user_interest_tb=user_interest_tb.reset_index(drop=True)
seewoedu_user_interest_model=pd.io.sql.to_sql(user_interest_tb,'seewoedu_user_interest_model',con=conn,if_exists='replace',index=user_interest_tb['$tid'])

