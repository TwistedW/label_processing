import os
import shutil

Agepath = "E:\\Python\\ageface\\ageface86\\86/"
Agepath1 = "E:\\Python\\ageface\\ageface86\\20-/"
Agepath2 = "E:\\Python\\ageface\\ageface86\\20-30/"
Agepath3 = "E:\\Python\\ageface\\ageface86\\30-40/"
Agepath4 = "E:\\Python\\ageface\\ageface86\\40-50/"
Agepath5 = "E:\\Python\\ageface\\ageface86\\50-60/"
Agepath6 = "E:\\Python\\ageface\\ageface86\\60+/"

list = os.listdir(Agepath)

for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if os.path.splitext(imgName)[1] != ".jpg":
        continue
    if i % 50==0:
        print(imgName)
    for j in range(0, len(imgName)):
        if imgName[j] == "_":
            for k in range(j+1, len(imgName)):
                if imgName[k] == "_":
                    age1 = imgName[k+1:k+5]
                    for n in range(k+1, len(imgName)):
                        if imgName[n] == "_":
                            age2 = imgName[n+1:n+5]
                            age = int(age2) - int(age1)
                            oldname = Agepath + imgName
                            if age < 20:
                                newname = Agepath1 + imgName
                                shutil.copy(oldname, newname)
                            elif 20 <= age < 30:
                                newname = Agepath2 + imgName
                                shutil.copy(oldname, newname)
                            elif 30 <= age < 40:
                                newname = Agepath3 + imgName
                                shutil.copy(oldname, newname)
                            elif 40 <= age < 50:
                                newname = Agepath4 + imgName
                                shutil.copy(oldname, newname)
                            elif 50 <= age < 60:
                                newname = Agepath5 + imgName
                                shutil.copy(oldname, newname)
                            elif age >= 60:
                                newname = Agepath6 + imgName
                                shutil.copy(oldname, newname)


