---
title: 基于streamlit的工厂设备监管系统
photo_path: https://github.com/Treedy2020/TreeDyStack/raw/master/mkdocs/docs/blogs/FactorViewer/
---
## 基础图纸
![草稿]({{photo_path}}/base.png)
## 概念
> 上位机（Host Computer, Master or Supervisor）是一个通常用于监控和控制生产线、工厂设备或工业自动化系统的计算机。上位机与下位机（被控制设备，如生产线上的设备、机器人等）相互配合，通过通信协议（如Modbus、CAN、以太网等）进行数据交换和指令传输。

上位机的主要职责包括：

1. 监控和控制：上位机可以实时监控下位机的状态和数据，并对其发送控制命令，如启动、停止、调整参数等。
2. 数据采集与处理：上位机可以收集下位机的数据，如传感器读数、设备状态等，并对这些数据进行处理、分析和存储。
3. 故障诊断与报警：上位机可以对下位机发出的故障或异常信号进行诊断，并采取相应措施，如发出报警、停止设备等。
4. 人机界面（HMI）：上位机通常提供一种人机界面，以便操作员和管理员监控系统状态、控制设备、查看数据报表等。
5. 通信：上位机负责与下位机以及其他相关系统（如数据库、ERP系统等）之间的通信，确保数据的准确传输和系统之间的协同工作。

!!! info
    上位机可以是一台工业计算机、PC、服务器甚至嵌入式设备，具体取决于所需的功能和性能。上位机软件可以使用各种编程语言（如C/C++、Python、Java等）和开发工具（如LabVIEW、WinCC、Ignition等）进行开发。



## 基础类

``` mermaid
classDiagram
    class 工厂设备监管系统{
    	- 工厂名称
    	- 位置
    	- 生产线（多条）
    }
    
    class 生产线{
    	- 生产线编号
    	- 产品类型
    	- 设备列表
    	- 传感器列表
    	+ 数据采集方法()
    	+	数据处理、存储、分析方法()
    	+ 报警方法()
    }
    
    class 设备{
    	- 名称
    	- 类型
    	- 状态
    	- 操作
		- 所在产线位置
    }
    
    class 传感器{
    	- 类型（温度传感器、压力传感器等）
    	- 状态
    	- 读数
		- 所在产线位置
    }
    

		工厂设备监管系统--|> 生产线: 包含
		生产线 --|> 传感器: 包含
		生产线 --|> 设备: 包含
		
		class User{
    	- 姓名
			- 职位
			- 权限
    }
    
    class 日志{
    	- 日志类型
    	- 日志时间
    	- 操作人员
    	+ 记录系统操作和事件()
    	+ 日志处理方法()
    }
    
    class 通讯{
    	- 系统通信
    	+ 连接数据库方法()
    	+ 断开连接方法()
    	+ 发送数据方法()
    	+ 接收数据方法()
    }
```


## 系统框架

```mermaid
C4Context

Container_Boundary(enterprise, "工厂设备监管系统") {
	Person(users, "操作员\管理者", "生产线上的操作者")
	SystemDb(db, "数据库", "存储设备数据, 传感器数据, 操作日志, 指令集合并记录系统事件日志")
	SystemDb(instructions, "指令集合", "不同类型的设备指令集合以及系统指令集合")

	Container_Boundary(line1, "生产线 1", "生产线 1") {
		Container(sensors1, "传感器列表1")
		Container(equipments1, "设备列表1")
	}
	Container_Boundary(line2, "生产线 2") {
		Container(sensors2, "传感器列表2")
		Container(equipments2, "设备列表2")
	}
	Container_Boundary(line3, "生产线 3") {
		Container(sensors3, "传感器列表3")
		Container(equipments3, "设备列表3")
	}
	Container_Boundary(line4, "生产线 k") {
		Container(added, "...")	
	}
}

```
