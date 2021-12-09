import pickle
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
from flask import Flask, request, render_template, redirect
import numpy as np 
import time

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('landing.html')

@app.route('/K-means', methods=['GET', 'POST'])
def KMeans():
    if request.method == 'POST':
        f = request.files['file']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path",basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("uplaod folder is ",filepath)
        f.save(filepath)

        img = cv2.imread(str(filepath))/255

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scale_percent = 20
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)  
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        img = resized
        vectorised = img.reshape((-1,3))
        img_df = pd.DataFrame(vectorised)
        img_df.rename(columns={0:'R', 1:'G', 2: 'B'}, inplace =True)

        centroids = img_df.sample(5)
        X = img_df

        k = 5
        diff = 1
        j=0

        while(abs(diff)>0.05):
            XD=X
            i=1
            for index1,row_c in centroids.iterrows():
                ED=[]
                print("Calculating distance")
                for index2,row_d in tqdm(XD.iterrows()):
                    d1=(row_c["R"]-row_d["R"])**2
                    d2=(row_c["G"]-row_d["G"])**2
                    d3=(row_c["B"]-row_d["B"])**2
                    d=np.sqrt(d1+d2+d3)
                    ED.append(d)
                X[i]=ED
                i=i+1

            C=[]
            print("Getting Centroid")
            for index,row in tqdm(X.iterrows()):
                min_dist=row[1]
                pos=1
                for i in range(k):
                    if row[i+1] < min_dist:
                        min_dist = row[i+1]
                        pos=i+1
                C.append(pos)
            X["Cluster"]=C
            centroids_new = X.groupby(["Cluster"]).mean()[["R","G", "B"]]
            if j == 0:
                diff=1
                j=j+1
            else:
                diff = (centroids_new['R'] - centroids['R']).sum() + (centroids_new['G'] - centroids['G']).sum() + (centroids_new['B'] - centroids['B']).sum()
                print(diff.sum())
            centroids = X.groupby(["Cluster"]).mean()[["R","G","B"]]
        
        centroids = centroids.to_numpy()
        labels = X["Cluster"].to_numpy()
        segmented_image = centroids[labels-1]
        segmented_image = segmented_image.reshape(img.shape)


        preds = 1

        cv2.imshow("k-means_segmentation", segmented_image)
        cv2.waitKey(0)
        return render_template('landing.html')

@app.route('/Ostu', methods=['GET', 'POST'])
def Ostu():
    if request.method == 'POST':
        f = request.files['file']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path",basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("uplaod folder is ",filepath)
        f.save(filepath)

        img = cv2.imread(str(filepath))
        img = cv2.resize(img, (800, 600))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('binary', img_gray)
        cv2.waitKey(0)

        histg = cv2.calcHist([img_gray], [0], None, [255], [0, 255])

        within = []
        for i in range(len(histg)):
            x,y = np.split(histg, [i])
            x1 = np.sum(x)/(img.shape[0]*img.shape[1])
            y1 = np.sum(y)/(img.shape[0]*img.shape[1])
            x2 = np.sum([j*t for j,t in enumerate(x)])/np.sum(x)
            y2 = np.sum([j*t for j,t in enumerate(y)])/np.sum(y)
            x3 = np.sum([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
            x3 = np.nan_to_num(x3)
            y3 = np.sum([(j-y2)**2*t for j,t in enumerate(y)])/np.sum(y)
            y3 = np.nan_to_num(y3)
            within.append(x1*x3 + y1*y3)
        min_index = np.argmin(within)
        print(min_index)

        ret, thresh = cv2.threshold(img_gray, min_index, 255, cv2.THRESH_BINARY)
        cv2.imshow('binary', thresh)
        cv2.waitKey(0)

        return render_template('landing.html')


if __name__ == '__main__':
    app.run()