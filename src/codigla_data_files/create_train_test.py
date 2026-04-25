# Python script to produce train and test directories
# from zip files - Feb 2026. Check updating

import numpy as np
import pandas as pd
import zipfile as zf
from pathlib import Path 
from sklearn.model_selection import train_test_split

def check_import():
    print("Version 0.0")

def getFileNames(zipFilePath):
    try:
        with zf.ZipFile(zipFilePath,'r') as zipFile:
            fileList=zipFile.namelist()
            return fileList
    except zf.BadZipFile as bzip:
        bzip.add_note("Caught in getFileNames")
        raise 
    except Exception as e:
        e.add_note("Caught in getFileNames")
        raise

def createDirectories(rootPath,classnames=["absent","present"]):
    try:
        if not len(classnames)>0:
            raise RuntimeError("List of class names requires at least one element")
        p=Path(rootPath)
        if not p.exists():
            raise FileExistsError("Root path for train test directories doesn't exist")
        trainp = p / 'train'
        testp = p / 'test'
        if not trainp.exists():
            trainp.mkdir()
        if not testp.exists():
            testp.mkdir()
        ltrain=[]
        ltest=[]
        for name in classnames:
            trainclassp = trainp / name
            testclassp = testp / name
            if not trainclassp.exists():
                trainclassp.mkdir()
            if not testclassp.exists():
                testclassp.mkdir()
            ltrain.append(trainclassp)
            ltest.append(testclassp)
        return (ltrain,ltest)
    except Exception as e:
        e.add_note("Caught in createDirectories")
        raise
        
def resample(X,y,balance=1.0):
    return X,y #TODO

def create_train_test(folder,zipFileNames,classNames,doSplit=True, splitRandomState=None, trainBalance=None, testBalance=None):
    try:
        if not len(zipFileNames)>0:
            raise RuntimeError("At least  one element is required in zip-file list")
        if not len(classNames)>0:
            raise RuntimeError("At least  one element is required in class name list")
        destination_folder=Path(folder)
        print(destination_folder)
        ltrain, ltest =createDirectories(destination_folder,classNames)
        print("Directorios creados")

        classIndex=0
        X=[]
        y=[]
        for zipName in zipFileNames:
            lnames=getFileNames(zipName)
            X=X + [[n,p] for n,p in zip([zipName]*len(lnames),lnames)]

            y=y+[classIndex]*len(lnames)
            classIndex = classIndex + 1
            classIndex = classIndex % len(classNames)
        X=np.array(X)
        y=np.array(y)
        if doSplit:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=splitRandomState)
            if trainBalance is not None:
                #Resample
                X_train,y_train=resample(X_train,y_train,trainBalance) 
            if testBalance is not None:
                #Resample test
                X_test,y_test=resample(X_test,y_test,testBalance)
        else:
            X_test=X
            y_test=y
            if testBalance is not None:
                X_test,y_test=resample(X_test,y_test,testBalance)
            X_train=[]
            y_train=[]

        train_annotations={'file_name':[],'name':[],'label':[]}
        test_annotations={'file_name':[],'name':[],'label':[]}


        #for xtrain,xtest,ytrain,ytest in zip(X_train,X_test,y_train,y_test):
        #Train data
        for xtrain,ytrain in zip(X_train,y_train):
            try:
                # Training data
                className=classNames[ytrain]
                destp=destination_folder / Path('train/'+className)
                fileName=Path(xtrain[1]).name
                if Path(destp / fileName).exists():
                    print("destp=",destp)
                    print("fileName=",fileName)
                    raise RuntimeError("Trying to extract an already existing file. Aborting")
                if not destp.exists():
                    raise RuntimeError("Class directory doesn't exist")
                train_annotations['file_name'].append(className+'/'+fileName)
                train_annotations['label'].append(ytrain)
                train_annotations['name'].append(fileName)

                with zf.ZipFile(xtrain[0],'r') as zipFile:
                    zipInfo=zipFile.getinfo(xtrain[1])
                    zipInfo.filename=fileName #Change extracted file name
                    zipFile.extract(xtrain[1],path=destp)
            
            except zf.BadZipFile as bzip:
                bzip.add_note("Caught in process")
                raise 
            except Exception as e:
                e.add_note("Caught in process")
                raise

        #Test data
        for xtest,ytest in zip(X_test,y_test):
            try:
                className=classNames[ytest]
                destp=destination_folder / Path('./test/'+className)
                fileName=Path(xtest[1]).name
                if Path(destp / fileName).exists():
                    print("destp=",destp)
                    print("fileName=",fileName)
                    raise RuntimeError("Trying to extract an already existing file. Aborting")
                if not destp.exists():
                    raise RuntimeError("Class directory doesn't exist")
                test_annotations['file_name'].append(className+'/'+fileName)
                test_annotations['label'].append(ytest)
                test_annotations['name'].append(fileName)
                with zf.ZipFile(xtest[0],'r') as zipFile:
                    zipInfo=zipFile.getinfo(xtest[1])
                    zipInfo.filename=fileName #Change extracted file name
                    zipFile.extract(xtest[1],path=destp)
 
            except zf.BadZipFile as bzip:
                bzip.add_note("Caught in process")
                raise 
            except Exception as e:
                e.add_note("Caught in process")
                raise
    
    except Exception as e:
        e.add_note("Caught in process")
        raise

    # Now pandas dataframes
    traindf=pd.DataFrame(train_annotations)
    traindf=traindf.sort_values('label')
    traindf.to_csv(destination_folder / Path('train/metadata.csv'),index=False)
    testdf=pd.DataFrame(test_annotations)
    testdf=testdf.sort_values('label')
    testdf.to_csv(destination_folder / Path('test/metadata.csv'),index=False)



#if __name__ == "__main__":
#    create_train_test(trial_0,["../../../npapila_real_normal.zip","../../../npapila_real_glaucoma.zip","../../../nrimone_real_normal.zip", "../../../nrimone_real_glaucoma.zip"],["normal","glaucoma"],doSplit=True, splitRandomState=100)
