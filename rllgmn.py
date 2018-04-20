import numpy
import math
from matplotlib import pyplot

class RLLGMN:

    ## 0 コンストラクタ
    def __init__(self, n_Data_size, n_Input_unit_size, n_Hidden_unit_size, n_Hidden_layer_size, n_Size_Output_unit):
        self.unit_5out=numpy.zeros((n_Size_Output_unit,1))
        self.unit_4
        #unit0=[numpy.zeros((self.unit_size[i],1)) for i in range(len(self.unit_size))]
        self.weight = [numpy.random.random_sample((self.unit_size[i2+1],self.unit_size[i2]+1)) for i2 in range(LayerSize_weight)]
        self.momentum = [numpy.random.random_sample((self.unit_size[i2+1],self.unit_size[i2]+1)) for i2 in range(LayerSize_weight)]
'''        
    ### public
    ## 1 学習
    def train(self, datas_input, datas_output_true, Epsilon, Mu, Epoch, Online):
        
        self.error = numpy.zeros(Epoch)
        for Count_epo in range(Epoch):
            if Online==1:
                for iy in range(datas_input.shape[0]):
                    onedata_input=datas_input[iy, :].reshape(1,-1)
                    onedata_output_true=datas_output_true[iy, :].reshape(1,-1)
                    err,datas_unit = self.__update_weight(onedata_input, onedata_output_true, Epsilon, Mu)
                    self.error[Count_epo]+=err
            else:
                self.error[Count_epo],datas_unit = self.__update_weight(datas_input, datas_output_true, Epsilon, Mu)
            '''    
            if (Count_epo+1) % (Epoch/10) ==0:
                print("学習率：",(Count_epo+1)*100 / Epoch,"%")
                print("誤差",self.error[Count_epo],end="\n\n")
            '''
        self.__error_graph()
        return datas_unit,self.weight
        
    
    ## 2 検証
    def predict(self, datas_input):
        DataSize = datas_input.shape[0]
        unit0=[numpy.zeros((self.unit_size[i],1)) for i in range(len(self.unit_size))]
        datas_unit = [unit0 for i in range(DataSize)]
        #print(datas_unit)
        for iy in range(DataSize):
            onedata_input=datas_input[iy, :].reshape(1,-1)
            onedata_unit = self.__forward(iy,onedata_input)# 1-1-1
            datas_unit[iy] = onedata_unit

        return datas_unit#(C, Y)


    ### private
    # 1-1 重みの更新 : 誤差逆伝搬法を用いている
    def __update_weight(self, datas_input, datas_output_true, Epsilon, Mu):
        DataSize = datas_input.shape[0]# =1
        err = 0.0
        unit0=[numpy.zeros((self.unit_size[i],1)) for i in range(len(self.unit_size))]
        datas_unit = [unit0 for i in range(DataSize)]
        
        # (1) 重みと入力から各ユニットの値を計算
        for iy in range(DataSize):
            onedata_input=datas_input[iy, :].reshape(1,-1)
            onedata_output_true=datas_output_true[iy,:].reshape(1,-1)
            onedata_unit = self.__forward(iy,onedata_input)# 1-1-1
            datas_unit[iy]=onedata_unit
            onedata_output=onedata_unit[len(self.unit_size)-1]
            err += numpy.dot( (onedata_output_true.reshape(1,-1) - onedata_output.reshape(1,-1)).reshape(1,-1), (onedata_output_true.reshape(1,-1) - onedata_output.reshape(1,-1)).reshape((-1, 1)) ) / 2.0
        # (2) 次の層
        olddelta=[[0] for i in range(DataSize)]
        for Count_layer in range(len(self.unit_size)-2,-1,-1):#4,3,2,1,0,
            sumdelta=numpy.zeros((self.weight[Count_layer].shape[0],self.weight[Count_layer].shape[1]))
            oldweight=self.weight[Count_layer]
            for Count_data in range(DataSize):
                #print(Count_data,"番目")
                if Count_layer==(len(self.unit_size)-2):#5
                    newdelta = (datas_unit[Count_data][Count_layer+1] - datas_output_true[Count_data, :].reshape(-1,1) )    * datas_unit[Count_data][Count_layer+1] * (1.0 - datas_unit[Count_data][Count_layer+1])
                else:
                    newdelta = numpy.dot(self.weight[Count_layer+1][:, 1:].T, olddelta[Count_data])                         * datas_unit[Count_data][Count_layer+1] * (1.0 - datas_unit[Count_data][Count_layer+1])
                sumdelta += newdelta.reshape((-1, 1)) * (numpy.r_[numpy.ones((1,1)), datas_unit[Count_data][Count_layer]]).T
                #self.weight[Count_layer] -= Epsilon * (newdelta.reshape((-1, 1)) * (numpy.r_[numpy.ones((1,1)), datas_unit[Count_data][Count_layer]]).T)
                #if Count_layer==2 :
                #    print(self.weight[Count_layer][0,0])
                olddelta[Count_data] = newdelta

                self.weight[Count_layer] -= Epsilon *sumdelta/DataSize #-0.9*self.momentum[Count_layer]
            
            self.momentum[Count_layer]=self.weight[Count_layer]-oldweight
        return err,datas_unit
    
    
    # 1-1-1 フォワード : 与えられた w,bを計算し,シグモイド関数を適応することで、各ユニット(入力,隠れ,出力ユニット)を算出する
    def __forward(self,DataNum , onedata_input):
        onedata_unit=[numpy.zeros((self.unit_size[i],1)) for i in range(len(self.unit_size))];
        onedata_unit[0]=onedata_input.reshape(-1,1)
        for Count_layer in range(len(self.unit_size)-1):#012345
            onedata_unit[Count_layer+1] = self.__sigmoid(   numpy.dot( self.weight[Count_layer], numpy.concatenate([numpy.ones((1,1)), onedata_unit[Count_layer]], axis=0) )   )
        return onedata_unit
    
    
    # 1-1-1-1 シグモイド関数
    def __sigmoid(self, arr):
        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)
    
    
    # 1-2 誤り改善状況の出力
    def __error_graph(self):
        print("\n\n\n")
        print("========================(2)誤り改善状況========================")
        pyplot.ylim(0.0, 2.0)
        pyplot.plot(numpy.arange(0, self.error.shape[0]), self.error)
        pyplot.show()
        print("=============================================================")
'''