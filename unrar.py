import rarfile


rar_name = 'merged_output.rar'


with rarfile.RarFile(rar_name) as opened_rar:
    opened_rar.extractall()

print("ok")
