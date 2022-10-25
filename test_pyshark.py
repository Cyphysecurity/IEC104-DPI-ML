import pyshark
from pyshark import FileCapture, LiveCapture

cap = FileCapture('single.pcap', display_filter='104apci')
#print(type(cap), cap[5])
#print(cap[12]['104ASDU'])
print(cap[0]['104ASDU'].QDS)
