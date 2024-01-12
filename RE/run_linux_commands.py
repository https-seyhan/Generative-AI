import os
stream = os.popen('ls -la')
output = stream.readlines()
print(output)
#ooo_cat USE_CASE_*.odt -o ALL_USE_CASES.odt

