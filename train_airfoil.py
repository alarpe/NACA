import os
import sys
import torch
# import torch_npu  # 如需使用华为NPU请取消注释
#from torch_npu.contrib import transfer_to_npu  #
import numpy as np
import pickle  ##
import matplotlib.pyplot as plt
from torch import optim,nn
from model import EncodeProcessDecode
from torch_geometric.data import DataLoader
# from data import get_mesh_graph #process_data
import math
from log import logger_setup
from data import MeshAirfoilDataset,log_images_1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#####
distributed = False
if distributed:
   local_rank = int(os.environ["LOCAL_RANK"])
   world_size= int(os.environ["WORLD_SIZE"])
   device = torch.device('cuda', local_rank)
   print(device)
   os.environ["MASTER_ADDR"] = "127.0.0.1"    # 用户需根据自己实际情况设置
   os.environ["MASTER_PORT"] = "29502"    # 用户需根据自己实际情况设置
   print(world_size)
   torch.distributed.init_process_group(backend="nccl",rank=local_rank, world_size=world_size)
#torch.cuda.set_device(device)

dataset=[]
program_path = r'd:\NACA'
train_loader=MeshAirfoilDataset(program_path + '/data2/',mode='train')
val_loader=MeshAirfoilDataset(program_path + '/data2/',mode='val')
#
in_channels = train_loader.get(0).x.shape[-1]
out_channels = train_loader.get(0).y.shape[-1]
hidden_channels = 128
proc_layers = 6
latent_size = 128
num_layers = 2
batchsize = 1

#
n_epoch = 150
case_folder = 'case1'
save_path = program_path +'/'+case_folder
if not os.path.exists(save_path):
    os.makedirs(save_path)
n_fig = 5
SEED_TEST = 8765
#
learning_rate = 0.0005
gamma = 0.5
epochs_sch = [70,100,110,120,130,140]
# scale_mesh: x_min,x_max,y_max
scale_mesh_train = [1.0,1.0,1.0]
scale_mesh_test = [1.0,1.0,1.0]
scale_random = [0.8,1.0]  
zoom = True
xb=[-1.5,2.5]
yb=[-2.0,2.0]
###
modelname = 'GAT' #'GCN'
beta = (0.8,0.9)
weight_decay = 1e-3
model_loading = False # default:False (training from scratch)
epoch_start = 0
if model_loading:
    loading_name = 'case1/model.pkl'
    loading_path = program_path +'/'+loading_name
epoch_int = 50
lambda_1 = 0.0
lambda_gnn = 0.8


print("a = ", train_loader.a,", b = ", train_loader.b)
for i in range(train_loader.len):
    data=train_loader.get(i)
    dataset.append(data)
    if i==0:
        print("i = 0, a = ", train_loader.a,", b = ", train_loader.b)
        print("aoa = ", data.aoa.numpy(), ", spd = ", data.spd.numpy()) ###

print("a = ", train_loader.a,", b = ", train_loader.b)
loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers = 4)

#
data=train_loader.get(0,mode=2)
print("edge_attr.shape = ",data.edge_attr.shape,", x.shape = ",data.x.shape, ", edge_index.shape = ",data.edge_index.shape,", y.shape = ",data.y.shape)
###
print("edge_attr:")
print(data.edge_attr)
print("range: ", np.amin(data.edge_attr.numpy()), ", ", np.amax(data.edge_attr.numpy()))
print("mean: ", np.mean(data.edge_attr.numpy()))
print("edge_index:")
print(data.edge_index)
print("x:")
print(data.x)
print("range: ", np.amin(data.x.numpy(),axis=0), ", ", np.amax(data.x.numpy(),axis=0))
print("y:")
print(data.y)
print("range: ", np.amin(data.y.numpy(),axis=0), ", ", np.amax(data.y.numpy(),axis=0))

#
print("elements len = ",len(data.elems_list))
is_triangle = True
for element in data.elems_list:
    if len(element)!=3:
        is_triangle = False
print("is_triangle: ",is_triangle)

# load validation set
if 1:
   dataset=[]
   for i in range(val_loader.len):
       data=val_loader.get(i)
       #data = process_data(data,scale_mesh_test)
       print("scale = ",scale_mesh_test,", x.shape = ",data.x.shape, ", edge_index.shape = ",data.edge_index.shape)
       dataset.append(data)
   if distributed:
       sampler = torch.utils.data.distributed.DistributedSampler(dataset)
       loader_val = DataLoader(dataset, sampler=sampler, batch_size=1)
   else:
       loader_val = DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 0)
   #root_logger.info("Validation data loaded")

root_logger = logger_setup(os.path.join(save_path, 'logairfoil.log'))
root_logger.info("===========parameters============")

mp = {}
if modelname=='GCN':
    mp['improved'] = False
    mp['cached'] = False
    mp['bias'] = True
    root_logger.info("improved = " + str(mp['improved']))
    root_logger.info("cached = " + str(mp['cached']))
    root_logger.info("bias = " + str(mp['bias']))
elif modelname=='GAT':
    mp['heads'] = 2
    mp['slope'] = 0.2
    mp['dropout'] = 0.0
    root_logger.info("heads = " + str(mp['heads']))
    root_logger.info("slope = " + str(mp['slope']))
    root_logger.info("dropout = " + str(mp['dropout']))
else:
    modelname='SAGE'
    mp['aggr'] = 'mean'
    mp['root_weight'] = True
    mp['project'] = False
    root_logger.info("aggr = " + str(mp['aggr']))
    root_logger.info("root_weight = " + str(mp['root_weight']))
    root_logger.info("project = " + str(mp['project']))
input_sizes = [data.x.shape[1], data.edge_attr.shape[1]]
model = EncodeProcessDecode(
            input_sizes,
            out_channels,
            latent_size,
            num_layers,
            proc_layers,
            hidden_channels,
            modelname,
            mp,
            lambda_gnn).to(device)


root_logger.info("model name: "+modelname)
if model_loading:
    model.load_state_dict(torch.load(loading_path))

if distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank,
                                                      find_unused_parameters = True)

optimizers = optim.AdamW(model.parameters(), lr=learning_rate, betas=beta, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma, last_epoch=-1)
criterion =torch.nn.MSELoss().to(device)
model.train()

root_logger.info("case name: "+case_folder)
root_logger.info("proc_layers = "+str(proc_layers))
root_logger.info("hidden_channels = "+str(hidden_channels))
root_logger.info("num_layers = "+str(num_layers))
root_logger.info("latent_size = "+str(latent_size))
root_logger.info("batchsize = "+str(batchsize))
root_logger.info("n_epoch = "+str(n_epoch)) 
root_logger.info("epochs_sch = "+str(epochs_sch[0])+", "+str(epochs_sch[1])+\
                 ", "+str(epochs_sch[2])+", "+str(epochs_sch[3])+", "+str(epochs_sch[4])+", "+str(epochs_sch[5]))
root_logger.info("learning_rate = "+str(learning_rate)) 
root_logger.info("gamma = "+str(gamma)) 
root_logger.info("n_epoch = "+str(n_epoch)) 

root_logger.info("scale_mesh_train= "+str(scale_mesh_train[0])+", "+str(scale_mesh_train[1])+", "+str(scale_mesh_train[2]))
root_logger.info("scale_mesh_test= "+str(scale_mesh_test[0])+", "+str(scale_mesh_test[1])+", "+str(scale_mesh_test[2]))
root_logger.info("scale_random= "+str(scale_random[0])+", "+str(scale_random[1]))

root_logger.info("optimizer: AdamW")
root_logger.info("beta = "+str(beta[0])+", "+str(beta[1]))
root_logger.info("weight_decay = "+str(weight_decay))

root_logger.info("epoch_start = "+str(epoch_start))
root_logger.info("model_loading: "+str(model_loading))
if model_loading:
    root_logger.info("loading_name: " + loading_name)

root_logger.info("epoch_int = "+str(epoch_int))
root_logger.info("lambda_1 = "+str(lambda_1))
root_logger.info("lambda_gnn = "+str(lambda_gnn))

root_logger.info("===========start train===========")
root_logger.info("a = "+str(train_loader.a[0]))  #### v5
root_logger.info("b = "+str(train_loader.b[0]))  ####

loss_history=[]
epoch_history=[]
sum_loss_best = None
epoch_best = None

for epoch in range(epoch_start,n_epoch):
   sum_loss=0
   sum_loss_1=0
   sum_loss_total=0
   i_batch = 0
   root_logger.info("Epoch"+str(epoch+1)) 

   ########
   dataset=[]
   for i in range(train_loader.len):
       data=train_loader.get(i)
       scale_mesh_1=np.random.uniform(scale_random[0],scale_random[1])
       #data = process_data(data,scale_mesh_train)
       if epoch==0: #
           print("scale = ",scale_mesh_train,", x.shape = ",data.x.shape, ", edge_index.shape = ",data.edge_index.shape)
       dataset.append(data)
   if distributed:
       sampler = torch.utils.data.distributed.DistributedSampler(dataset)
       loader = DataLoader(dataset, sampler=sampler, batch_size=batchsize)
   else:
       loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers = 0)
   root_logger.info("Data loaded") 
   ########  

   for batch in loader:
      i_batch += 1
      truefield=batch.y.to(device)
      prefield=model(batch.to(device))
      mes_loss=criterion(prefield,truefield)
      mes_loss_total = mes_loss
      optimizers.zero_grad()
      mes_loss_total.backward()
      optimizers.step()
      sum_loss+=mes_loss.item()
      sum_loss_total += mes_loss_total.item()
   loss_history.append(sum_loss/len(loader))
   epoch_history.append(epoch)
   sum_loss_m=(sum_loss)/len(loader)
   sum_loss_total_m = (sum_loss_total) / len(loader)
   root_logger.info("        loss: " + str(sum_loss_m))
   root_logger.info("        loss_total:" + str(sum_loss_total_m))
   #
   if epoch+epoch_int>=n_epoch:
       sum_loss = 0.0
       for batch in loader_val:
           i_batch += 1
           batch = batch.to(device)
           truefield = batch.y  #
           ###
           prefield = model(batch)  #
           mes_loss = criterion(prefield, truefield)
           loss = mes_loss
           sum_loss += loss.item()
       sum_loss2 = (sum_loss) / len(loader)
       root_logger.info("        val_loss")
       root_logger.info("        " + str(sum_loss2))
       if sum_loss_best==None or sum_loss2<=sum_loss_best:
           epoch_best = epoch
           sum_loss_best = sum_loss2
           torch.save(model.state_dict(), save_path + '/model.pkl')
           root_logger.info("params saved")
   if((epoch==epochs_sch[0])|(epoch==epochs_sch[1])|(epoch==epochs_sch[2])|\
           (epoch==epochs_sch[3])|(epoch==epochs_sch[4])|(epoch==epochs_sch[5])):
      scheduler.step()
      torch.save(model.state_dict(),save_path + '/model_'+str(epoch)+'.pkl')
torch.save(model.state_dict(),save_path + '/model_last.pkl')
root_logger.info("epoch_best = " + str(epoch_best+1))
root_logger.info("sum_loss_best = " + str(sum_loss_best))

# plot loss history
n_loss = len(loss_history)
loss_history_array = np.zeros((2,n_loss))
loss_history_array[0,:] = epoch_history[:]
loss_history_array[1,:] = loss_history[:]
with open(save_path + '/loss_t.pkl','wb') as f:
    pickle.dump(loss_history_array,f)

plt.figure(dpi=200,figsize=(9,6))
plt.plot(epoch_history,loss_history,'-')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(save_path + '/loss_t.png')

##########################
# test
##########################

model.load_state_dict(torch.load(save_path + '/model.pkl'))
root_logger.info("===========start test===========") 
test_loader=MeshAirfoilDataset(program_path + '/data2/',mode='test')
dataset_test=[]
mode_data = 2

for i in range(test_loader.len):
    data=test_loader.get(i,mode=mode_data)
    #data = process_data(data,scale_mesh_test,mode=mode_data)
    if i<5:
        print("scale = ",scale_mesh_test,", x.shape = ",data.x.shape, ", edge_index.shape = ",data.edge_index.shape)
    dataset_test.append(data)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True, generator = torch.Generator().manual_seed(SEED_TEST))
model.eval()
with torch.no_grad():
     sum_loss=0
     i_batch = 0
     for batch in test_loader:
         i_batch += 1
         batch=batch.to(device)
         truefield=batch.y
         prefield=model(batch)
         mes_loss=criterion(prefield,truefield)
         loss=mes_loss.cpu()
         sum_loss+=loss.item()
         if i_batch <= n_fig:
             log_images_1(batch.pos, prefield, truefield, batch.batch, batch.elems_list[0], 'test',
                          file=case_folder + '/sample_' + str(i_batch) + '_', zoom=zoom, xb=xb, yb=yb)
             batch = batch.cpu()
             root_logger.info("batch i = " + str(i_batch))
             root_logger.info("aoa = " + str(batch.aoa.numpy()[0]))
             root_logger.info("spd = " + str(batch.spd.numpy()[0]))
             root_logger.info("loss =  " + str(loss.item()))

sum_loss_m=sum_loss/(len(test_loader))
root_logger.info("        test_loss")
root_logger.info("        " + str(sum_loss_m))


