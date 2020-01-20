# SQL

## JOIN

![](../../.gitbook/assets/sql-join.png)

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

## Source

{% embed url="https://www.cnblogs.com/cc11001100/p/9043946.html" caption="" %}

{% embed url="https://www.runoob.com/mysql/mysql-functions.html" caption="" %}

