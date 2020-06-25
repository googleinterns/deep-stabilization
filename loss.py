import torch
import numpy as np
from torch.autograd import Variable
import operator
import torch.nn.functional as F
from gyro.gyro_function import torch_QuaternionProduct, torch_QuaternionReciprocal

class verloss(torch.nn.Module):
    def __init__(self):
        super(verloss, self).__init__()
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    def forward(self, y_pred,inp1,inp2,inp2_back,inp7,inp6,inp77, nframe, batchsize,C,C_back,reg,P1,P2):#,P1,P2   
        y_pred1=Variable(torch.zeros((batchsize,nframe-1,inp1.size()[2],inp1.size()[3])).cuda(),requires_grad=False)
        y_pred2=Variable(torch.zeros((batchsize,inp1.size()[2],inp1.size()[3])).cuda(),requires_grad=False)
        inp5=Variable(torch.zeros((batchsize,9,nframe+1)).cuda(),requires_grad=False)
        y_pred1[:,:,16*4:16*4+y_pred.size()[2],16*4:16*4+y_pred.size()[3]]=y_pred
        y_pred2[:,16*4:16*4+y_pred.size()[2],16*4:16*4+y_pred.size()[3]]=inp6
        inp5[:,:,0]=inp77
        inp5[:,:,1:nframe]=inp7.view(batchsize,9,nframe-1)
        
        r0=Variable(torch.zeros((1,nframe)).cuda(),requires_grad=False)
        for i in range(0,nframe):
            if i==0:
                tempY=torch.FloatTensor(1,1,y_pred1.size()[2],y_pred1.size()[3]).cuda()
                tempY[0,0,:,:]=y_pred1[:,i,:,:]
                tempC=torch.FloatTensor(1,y_pred1.size()[2],y_pred1.size()[3],2).cuda()
                tempC[:,:,:,1]=(C[:,:,:,2*i]-(y_pred1.size()[2]/2))/(y_pred1.size()[2]/2)
                tempC[:,:,:,0]=(C[:,:,:,2*i+1]-(y_pred1.size()[3]/2))/(y_pred1.size()[3]/2)
                temp5=F.grid_sample(tempY,tempC)

                temprow=Variable(torch.zeros((2,batchsize)).cuda(),requires_grad=False)
                for j in range(0,batchsize):
                    temprow[0,j]=torch.mean((((inp5[j,0,i]+1)*inp1[j,1,:,:]+inp5[j,3,i]*inp1[j,0,:,:]+inp5[j,6,i]+y_pred2)
                    -((inp5[j,0,i+1]+1)*inp2[j,2*i+1,:,:]+inp5[j,3,i+1]*inp2[j,2*i,:,:]+inp5[j,6,i+1]+temp5[j,0,:,:]))**2)
                    temprow[1,j]=torch.mean((((inp5[j,0,i]+1)*inp2_back[j,2*i+1,:,:]+(inp5[j,3,i])*inp2_back[j,2*i,:,:]+inp5[j,6,i])
                    -((inp5[j,0,i+1]+1)*inp1[j,1,:,:]+(inp5[j,3,i+1])*inp1[j,0,:,:]+inp5[j,6,i+1]+y_pred1[j,i,:,:]))**2)
                    
            elif i==nframe-1:
            
                tempY=torch.FloatTensor(1,1,y_pred1.size()[2],y_pred1.size()[3]).cuda()
                tempY[0,0,:,:]=y_pred1[:,i-1,:,:]
                tempC=torch.FloatTensor(1,y_pred1.size()[2],y_pred1.size()[3],2).cuda()
                tempC[:,:,:,1]=(C_back[:,:,:,2*i]-(y_pred1.size()[2]/2))/(y_pred1.size()[2]/2)
                tempC[:,:,:,0]=(C_back[:,:,:,2*i+1]-(y_pred1.size()[3]/2))/(y_pred1.size()[3]/2)
                temp6=F.grid_sample(tempY,tempC)
                
                temprow=Variable(torch.zeros((2,batchsize)).cuda(),requires_grad=False)
                for j in range(0,batchsize):
                    temprow[0,j]=torch.mean((((inp5[j,0,i]+1)*inp1[j,1,:,:]+inp5[j,3,i]*inp1[j,0,:,:]+inp5[j,6,i]+y_pred1[j,i-1,:,:])
                    -((inp5[j,0,i+1]+1)*inp2[j,2*i+1,:,:]+inp5[j,3,i+1]*inp2[j,2*i,:,:]+inp5[j,6,i+1]))**2)
                    temprow[1,j]=torch.mean((((inp5[j,0,i]+1)*inp2_back[j,2*i+1,:,:]+(inp5[j,3,i])*inp2_back[j,2*i,:,:]+inp5[j,6,i]+temp6[j,0,:,:])
                    -((inp5[j,0,i+1]+1)*inp1[j,1,:,:]+(inp5[j,3,i+1])*inp1[j,0,:,:]+inp5[j,6,i+1]))**2)
                
            else:

                tempY=torch.FloatTensor(1,1,y_pred1.size()[2],y_pred1.size()[3]).cuda()
                tempY[0,0,:,:]=y_pred1[:,i,:,:]
                tempC=torch.FloatTensor(1,y_pred1.size()[2],y_pred1.size()[3],2).cuda()
                tempC[:,:,:,1]=(C[:,:,:,2*i]-(y_pred1.size()[2]/2))/(y_pred1.size()[2]/2)
                tempC[:,:,:,0]=(C[:,:,:,2*i+1]-(y_pred1.size()[3]/2))/(y_pred1.size()[3]/2)
                temp5=F.grid_sample(tempY,tempC)

                tempY=torch.zeros(1,1,y_pred1.size()[2],y_pred1.size()[3]).cuda()
                tempY[0,0,:,:]=y_pred1[:,i-1,:,:]
                tempC=torch.FloatTensor(1,y_pred1.size()[2],y_pred1.size()[3],2).cuda()
                tempC[:,:,:,1]=(C_back[:,:,:,2*i]-(y_pred1.size()[2]/2))/(y_pred1.size()[2]/2)
                tempC[:,:,:,0]=(C_back[:,:,:,2*i+1]-(y_pred1.size()[3]/2))/(y_pred1.size()[3]/2)
                temp6=F.grid_sample(tempY,tempC)

                temprow=Variable(torch.zeros((2,batchsize)).cuda(),requires_grad=False)
                
                for j in range(0,batchsize):
                    temprow[0,j]=torch.mean((((inp5[j,0,i]+1)*inp1[j,1,:,:]+inp5[j,3,i]*inp1[j,0,:,:]+inp5[j,6,i]+y_pred1[j,i-1,:,:])
                    -((inp5[j,0,i+1]+1)*inp2[j,2*i+1,:,:]+inp5[j,3,i+1]*inp2[j,2*i,:,:]+inp5[j,6,i+1]+temp5[j,0,:,:]))**2)
                    temprow[1,j]=torch.mean((((inp5[j,0,i]+1)*inp2_back[j,2*i+1,:,:]+(inp5[j,3,i])*inp2_back[j,2*i,:,:]+inp5[j,6,i]+temp6[j,0,:,:])
                    -((inp5[j,0,i+1]+1)*inp1[j,1,:,:]+(inp5[j,3,i+1])*inp1[j,0,:,:]+inp5[j,6,i+1]+y_pred1[j,i,:,:]))**2)      

            r0[0,i]=torch.mean(temprow)#+torch.mean(tempcol)
        pen1=torch.mean(r0)
        
        return pen1 

class C2_Smooth_loss(torch.nn.Module):
    def __init__(self):
        super(C2_Smooth_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, Qt, Qt_1, Qt_2):
        # Mt = torch_ConvertQuaternionToRotationMatrix(Qt)
        # Mt_1 = torch_ConvertQuaternionToRotationMatrix(Qt_1)
        # Mt_2 = torch_ConvertQuaternionToRotationMatrix(Qt_2)
        # detaQt = torch_ConvertRotationMatrixToQuaternion(torch.matmul(Mt, torch.inverse(Mt_1)))
        # detaQt_1 = torch_ConvertRotationMatrixToQuaternion(torch.matmul(Mt_1, torch.inverse(Mt_2)))
        # detaQ = Qt * Qt_1^-1
        detaQt = torch_QuaternionProduct(Qt, torch_QuaternionReciprocal(Qt_1))  
        detaQt_1 = torch_QuaternionProduct(Qt_1, torch_QuaternionReciprocal(Qt_2))  
        return self.MSE(detaQt, detaQt_1)
