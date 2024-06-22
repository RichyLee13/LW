def read_columns_to_dict(file_path):
    column_dict = {}

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行尾的换行符并分割成列
                columns = line.strip().split('\t')

                # 检查列数是否足够
                if len(columns) >= 3:
                    # 将第一列作为键，第二列和第三列作为值的元组
                    key = columns[0]
                    value = (columns[1], columns[2])

                    # 将键值对添加到字典中
                    column_dict[key] = value
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")

    return column_dict


# 示例使用
# file_path = 'datasets/Dataset_mask/statistics.txt'  # 请替换为你的文件路径
# data_dict = read_columns_to_dict(file_path)
#
# # 打印结果
# for key, value in data_dict.items():
#     print(f"Key: {key}, Value: {value}")
#
# # 根据第一列的内容查找第二列和第三列的内容
# print(data_dict['00001'])  # 输出：('Air', 'LWIR')