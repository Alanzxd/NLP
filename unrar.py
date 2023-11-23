import rarfile

# RAR文件的名称
rar_name = 'merged_output.rar'

# 使用rarfile解压
with rarfile.RarFile(rar_name) as opened_rar:
    opened_rar.extractall()

print("解压完成")
