"""Raise only this offline viewer's own X11 window and capture its content."""
import ctypes as C,json,re,subprocess,time
from pathlib import Path
from PIL import ImageGrab
ROOT=Path(__file__).resolve().parent
pid=json.loads((ROOT/'viewer_process.json').read_text())['pid']
lines=subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines()
window=None
for line in lines:
 if 'MuJoCo' not in line:continue
 match=re.search(r'0x[0-9a-fA-F]+',line)
 if not match:continue
 w=match.group()
 props=subprocess.check_output(['xprop','-id',w,'_NET_WM_PID'],text=True)
 if re.search(r'=\s*'+str(pid)+r'\s*$',props):window=int(w,16);break
assert window is not None,'Own MuJoCo window not found'
x=C.CDLL('libX11.so.6');x.XOpenDisplay.argtypes=[C.c_char_p];x.XOpenDisplay.restype=C.c_void_p
x.XDefaultRootWindow.argtypes=[C.c_void_p];x.XDefaultRootWindow.restype=C.c_ulong
x.XInternAtom.argtypes=[C.c_void_p,C.c_char_p,C.c_int];x.XInternAtom.restype=C.c_ulong
class Data(C.Union):_fields_=[('l',C.c_long*5),('b',C.c_char*20)]
class Event(C.Structure):_fields_=[('type',C.c_int),('serial',C.c_ulong),('send_event',C.c_int),('display',C.c_void_p),('window',C.c_ulong),('message_type',C.c_ulong),('format',C.c_int),('data',Data),('_padding',C.c_byte*96)]
x.XSendEvent.argtypes=[C.c_void_p,C.c_ulong,C.c_int,C.c_long,C.c_void_p];x.XFlush.argtypes=[C.c_void_p]
d=x.XOpenDisplay(b':0');assert d
msg=Event();msg.type=33;msg.send_event=1;msg.display=d;msg.window=window
msg.message_type=x.XInternAtom(d,b'_NET_ACTIVE_WINDOW',0);msg.format=32;msg.data.l[0]=2
x.XSendEvent(d,x.XDefaultRootWindow(d),0,(1<<20)|(1<<19),C.byref(msg));x.XFlush(d)
time.sleep(.5)
active=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)
assert int(re.search(r'0x[0-9a-fA-F]+',active).group(),16)==window,'Own viewer not active'
info=subprocess.check_output(['xwininfo','-id',hex(window)],text=True)
def value(name):return int(re.search(re.escape(name)+r':\s*(-?\d+)',info).group(1))
left,top,width,height=[value(k) for k in ['Absolute upper-left X','Absolute upper-left Y','Width','Height']]
ImageGrab.grab(bbox=(left,top,left+width,top+height),xdisplay=':0').save(ROOT/'viewer_window.png')
report=dict(pid=pid,window=hex(window),bbox=[left,top,left+width,top+height],own_viewer_active=True,hardware_connected=False)
(ROOT/'window_status.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
