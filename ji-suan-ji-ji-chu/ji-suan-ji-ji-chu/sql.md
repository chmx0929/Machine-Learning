# SQL

## JOIN

![](../../.gitbook/assets/sql-join.png)

## 建表

```text
CREATE TABLE IF NOT EXISTS tableName(columnName string comment '列注释') PARTITIONED BY (dt string);
INSERT OVERWRITE TABLE tableName PARTITION(dt=20200214)
SELECT
```

```text
CREATE TABLE tableName(
    columnName1 string comment '列1注释',
    columnName2 string comment '列2注释')
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'

LOAD DATA LOCAL INPATH 'data.csv' OVERWRITE INTO TABLE tableName
```

## collect\_list/collect\_set

![](../../.gitbook/assets/hive%20%281%29.png)

### collect\_list：不去重直接成list\(李四看了两次霸王别姬\)

```text
SELECT
      username
    , collect_list(video_name)
FROM
    t_visit_video GROUP BY username
```

![](../../.gitbook/assets/collect_list.png)

### collect\_set：进行去重\(李四看过霸王别姬\)

```text
SELECT
      username
    , collect_set(video_name)
FROM
      t_visit_video group by username
```

![](../../.gitbook/assets/collect_set.png)

## concat/concat\_ws

| 命令 | 说明 | 举例 | 常用 |
| :---: | :---: | :---: | :---: |
| CONCAT\(s1,s2...sn\) | 字符串 s1,s2 等多个字符串合并为一个字符串 | SELECT CONCAT\("SQL ", "Runoob ", "Gooogle ", "Facebook"\) AS ConcatenatedString | SELECT CONCAT\(collect\_set\(video\_name\)\) AS ConcatenatedString |
| CONCAT\_WS\(x, s1,s2...sn\) | 同 CONCAT\(s1,s2,...\) 函数，但是每个字符串之间要加上 x，x 可以是分隔符 | SELECT CONCAT\_WS\("-", "SQL", "Tutorial", "is", "fun!"\)AS ConcatenatedString; | SELECT CONCAT\_WS\(",",collect\_set\(video\_name\)\) AS ConcatenatedString |

## split

split\(str, regex\) - Splits str around occurances that match regex

split\('a,b,c,d',','\) 得到的结果：\["a","b","c","d"\]

split\('a,b,c,d',','\)\[0\] 得到的结果：a

## 数组转行

```text
SELECT ID,itemsName,name,loc
FROM Table
LATERAL VIEW explode(items) itemTable AS itemsName;
```

```text
ID   |    items                                  | name  |  loc  
_________________________________________________________________

id1  | ["item1","item2","item3","item4","item5"] | Mike | CT
id2  | ["item3","item7","item4","item9","item8"] | Chris| MN
```

```text
ID   |    items                       | name  |  loc  
______________________________________________________
id1  | item1                          | Mike  | CT
id1  | item2                          | Mike  | CT
id1  | item3                          | Mike  | CT
id1  | item4                          | Mike  | CT
id1  | item5                          | Mike  | CT
id2  | item3                          | Chris | MN
id2  | item7                          | Chris | MN
id2  | item4                          | Chris | MN
id2  | item9                          | Chris | MN
id2  | item8                          | Chris | MN
```



## 各种问题

### 内存溢出

```text
SET yarn.app.mapreduce.am.resource.mb=4096;
SET yarn.app.mapreduce.am.command-opts=-Xmx4000m;
```

### 动态分区

```text
SET hive.exec.dynamic.partition.mode=nonstrict;
```

### 数据压缩

```text
SET hive.exec.compress.intermediate=true;
SET mapreduce.map.output.compress=true;
SET mapred.map.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec;
SET mapred.map.output.compression.codec=com.hadoop.compression.lzo.LzoCodec;
SET hive.exec.compress.output=true;
SET mapred.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec;
```

### 晚起reduce

```text
SET mapreduce.job.reduce.slowstart.completedmaps=0.9;
```

### 增加reduce

```text
SET mapred.reduce.tasks=1000;
```

### 不限制数据分块数

```text
SET mapreduce.jobtracker.split.metainfo.maxsize=-1;
```

## Source

{% embed url="https://www.cnblogs.com/cc11001100/p/9043946.html" caption="" %}

{% embed url="https://www.runoob.com/mysql/mysql-functions.html" caption="" %}

