import laser
#import aestream
l = laser.Laser()
l.on()
y_base = 500
x_base = 500
square_shift = 500
nb_repetitions = 10
for i in range(10):
    for x in range(square_shift):
        l.move(x_base+x, y_base)
    for y in range(square_shift):
        l.move(x_base+square_shift,y_base+y)
    for x in range(square_shift):
        l.move(x_base+square_shift-x,y_base+square_shift)
    for y in range(square_shift):
        l.move(x_base,y_base+square_shift-y)
l.off()


