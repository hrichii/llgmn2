import numpy
import math
from matplotlib import pyplot
from rllgmn import *

if __name__ == '__main__':
    ##1 データセット
    #1-1 教師データ
    #1-1-1 入力
    
    #learn_in = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    #learn_out = numpy.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    #'''
    learn_in = numpy.array([[0, 0, 0],
                            [1, 0, 1],
                            [1, 1, 1], 
                            [1, 1, 0],
                            [1, 0, 0], 
                            [0, 0, 1],                    
                            ])
    #1-1-2 出力
    learn_out = numpy.array([[0],
                             [0],
                             [1],
                             [0],
                             [1],
                             [1]
                             ])
    #'''
    #1-2 検証データ
    #1-2-1 入力
    #'''
    test_in = numpy.array([[0, 1, 0],
                           [0, 1, 1],
                           ])
    #1-2-2 出力
    test_out = numpy.array([[1],
                            [0],
                            ])
    #'''
    #test_in=learn_in
    #test_out=learn_out
    
    ##2 条件入力
    while True:
        Unit="2";
        Layer="1";
        Online="1";
        print("=====================(1)数字を入力してください=====================")
        print("中間層のユニット数を入力")
        #Unit=input(">>>")
        print("中間層の層数を入力")
        #Layer=input(">>>")
        print("学習枠組みを選択(0:バッチ学習　1:オンライン学習)")
        #Online=input(">>>")

        if Unit.isdigit()*Layer.isdigit()*Online.isdigit()==1:
            Unit=int(Unit)
            Layer=int(Layer)
            Online=int(Online)
            if Unit*Layer==0:
                print("再入力\n\n\n")
            else:
                break
        else:
            print("再入力\n\n\n")
    
    
    ##3 パラメータ設定
    Date_num = learn_in.shape[0]                # データ数
    Input_unit_size = learn_in.shape[1]         # 入力ユニット数
    Hidden_unit_size = Unit                     # 隠れ層のユニット数
    Hidden_layer_size = Layer                   # 隠れ層数
    Output_unit_size = learn_out.shape[1]       # 出力ユニット数
    Epsilon = 0.1                               #　学習率
    Mu = 0.9                                    # モーメンタム係数
    Epoch = 1000                                   # 学習回数
    #Online                                     # 学習の枠組み(0:バッチ, 1:オンライン)
    maru=numpy.full((max([Input_unit_size,Hidden_unit_size,Output_unit_size]),2+Hidden_layer_size),"　　　")   #ユニットの構造
    ix=0
    for iy in range(Input_unit_size):
        maru[iy,ix]=str("〇")
        
    for ix in range(1,Hidden_layer_size+1):
        for iy in range(Hidden_unit_size):
            maru[iy,ix]="〇"
        
    ix=Hidden_layer_size + 1;
    for iy in range(Output_unit_size):
        maru[iy,ix]="〇";
    
    

    ##4 処理内容出力
    #学習枠組み出力
    if Online==1:
        print("-------------------------オンライン学習-------------------------")

    else:
        print("-------------------------バッチ学習-------------------------")
    
    #ラベル出力 
    label=(["中"]*(2+Hidden_layer_size))
    label[0]="入"
    label[2+Hidden_layer_size-1]="出"

    for ix in range(len(label)):
        print(label[ix].center(4),end='')
    print("\n".rjust(4),end='')

    #ユニット出力
    for iy in range(maru.shape[0]):
        for ix in range(maru.shape[1]):
            print(maru[iy,ix].center(4),end='')
        print("\n".rjust(4),end='')
    print("-------------------------------------------------------------")

    print("=============================================================")

    
    
    ##5 NN生成
    #入力するデータに関してバッチ学習を行う
    if Online==1:
        Data_size=1
    else:
        Data_Size=learn_in.shape[0]
    nn = RLLGMN(Data_size, Input_unit_size, Hidden_unit_size, Hidden_layer_size, Output_unit_size)
    
    ##6 学習
    trained_unit,trained_weight=nn.train(learn_in, learn_out, Epsilon, Mu, Epoch, Online)
    error=nn.error


    ##7 検証
    datas_unit=nn.predict(test_in)
    
    
    
    ##8 結果の出力
    datas_out=numpy.zeros((test_out.shape[0],test_out.shape[1]))
    for i in range(test_out.shape[0]):
        datas_out[i]=datas_unit[i][Hidden_layer_size+1].reshape(1,-1)
    
    print("\n\n\n")
    print("=========================(3)検証結果==========================")
    for i in range(test_out.shape[0]):
        print("",end="\n")
        print(i+1,"番目の検証データとの比較")
        print("test_out",end="")
        print("    ",end="")
        print("NN_out")
        print(test_out[i,:],end="")
        print("    ",end="")
        print(datas_out[i,:],end="")
        print("",end="\n")
    print("==============================================================")