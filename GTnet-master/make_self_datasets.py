import os
path_1 = "/home/xww/桌面/FSC/Aluminum"
save_path = '/home/xww/桌面/FSC/train.txt'
file = open(save_path, 'w')
images_file = open('/home/xww/桌面/FSC/trian.txt', 'w')
images_classes_file = open('/home/xww/桌面/FSC/train.txt', 'w')
a = 0
for i, name in enumerate(os.listdir(path_1)):
    # make classes.txt
    file.writelines(str(i+1) + " " + str(name) + '\n')
    c = os.listdir(path_1 + '/' + name)
    c.sort(key=lambda x: int(x.replace(name + "_", "").split('.')[0]))


    # make images.txt
       #os.listdir(path_1 + '/' + name).sort(key=lambda x:int(x[:-4]))

    for j, name1 in enumerate(c):
        a = a + 1
        print(">>>>>>>>*************>>>>>>>>>>", name1, j+1, a)
        #images_file.writelines(str(a) + " " + 'Aluminum/' + str(name1) + '\n')
        images_file.writelines(  'Aluminum/' + str(name1)+ " "+ str(a)  + '\n')
        images_classes_file.writelines(str(a) + ' ' + str(i+1) + '\n')
"""
c=os.listdir(path_1 + '/' + 'PS')
print(c)
c.sort(key=lambda x:int(x.replace("PS_", "").split('.')[0]))
print(c)
"""