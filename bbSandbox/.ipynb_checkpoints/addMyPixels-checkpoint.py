
import laser
import aestream
l = laser.Laser()
with aestream.USBInput((640,480)) as camera:
  while True:
    f = camera.read()
    F = 0;
    for i in range(1000):
      F = F + f.sum()  
      if F > 50000:
         l.on()
         F = 0
      else:
         l.off()


