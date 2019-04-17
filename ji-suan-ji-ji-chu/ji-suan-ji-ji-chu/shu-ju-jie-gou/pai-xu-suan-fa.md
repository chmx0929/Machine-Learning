# 排序算法

## **算法分类**

### **线性非线性**

**非线性时间比较类排序**：通过比较来决定元素间的相对次序，由于其时间复杂度不能突破O\(nlogn\)，因此称为非线性时间比较类排序。

**线性时间非比较类排序**：不通过比较来决定元素间的相对次序，它可以突破基于比较排序的时间下界，以线性时间运行，因此称为线性时间非比较类排序。 

![](../../../.gitbook/assets/849589-20180402132530342-980121409.png)

### 复杂度

![](../../../.gitbook/assets/849589-20180402133438219-1946132192.png)

### 相关概念

**稳定**：如果a原本在b前面，而a=b，排序之后a仍然在b的前面。

**不稳定**：如果a原本在b的前面，而a=b，排序之后 a 可能会出现在 b 的后面。

**时间复杂度**：对排序数据的总的操作次数。反映当n变化时，操作次数呈现什么规律。

**空间复杂度：**是指算法在计算机内执行时所需存储空间的度量，它也是数据规模n的函数。 

## 冒泡排序（Bubble Sort）

![](../../../.gitbook/assets/849589-20171015223238449-2146169197.gif)

## 选择排序（Selection Sort）

![](../../../.gitbook/assets/849589-20171015224719590-1433219824.gif)

## 插入排序（Insertion Sort）

![](../../../.gitbook/assets/cha-ru-pai-xu.gif)

## 希尔排序（Shell Sort）

![](../../../.gitbook/assets/xi-er-pai-xu-shell-sort.gif)

## 归并排序（Merge Sort）

![](../../../.gitbook/assets/gui-bing-pai-xu-merge-sort.gif)

```python
def merge(left, right):
    res = []
    while left and right:
        if left[0] < right[0]:
            res.append(left.pop(0))
        else:
            res.append(right.pop(0))
    res = res + left + right
    return res

def mergesort(lists):
    if len(lists) <= 1:
        return lists
    mid = len(lists)//2
    left = mergesort(lists[:mid])
    right = mergesort(lists[mid:])
    return merge(left,right)
```

## 快速排序（Quick Sort）

![](../../../.gitbook/assets/kuai-su-pai-xu-quick-sort.gif)

```python
def partition(arr, beg, end):
    pivot = arr[end-1]
    i = beg - 1
    for j in range(beg, end-1):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[end-1] = arr[end-1], arr[i+1]
    return i + 1
 
def quicksort(arr, beg, end):
    if beg < end - 1:
        q = partition(arr, beg, end)
        quicksort(arr, beg, q)
        quicksort(arr, q+1, end)
```

## 堆排序（Heap Sort）

![](../../../.gitbook/assets/dui-pai-xu-heap-sort.gif)

## 计数排序（Counting Sort）

![](../../../.gitbook/assets/ji-shu-pai-xu-counting-sort.gif)

## 桶排序（Bucket Sort）

![](../../../.gitbook/assets/tong-pai-xu-bucket-sort.png)

## 基数排序（Radix Sort）

![](../../../.gitbook/assets/ji-shu-pai-xu-radix-sort.gif)

## Source

{% embed url="https://www.cnblogs.com/onepixel/p/7674659.html" %}



