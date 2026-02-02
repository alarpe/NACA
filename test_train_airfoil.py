import os
import sys
import torch
# import torch_npu  # 如需使用华为NPU请取消注释
#from torch_npu.contrib import transfer_to_npu
import numpy as np
from torch import optim,nn
import pickle
from model import EncodeProcessDecode
from torch_geometric.data import DataLoader
# from data import get_mesh_graph #process_data
import time
import math
from log import logger_setup
from data import MeshAirfoilDataset,log_images_1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(device)
dataset=[]
program_path = r'd:\NACA'

#
hidden_channels = 128
proc_layers = 6
latent_size = 128
num_layers = 2
batchsize = 1

case_folder = 'case1'
result_folder = 'fig1'
save_path = program_path +'/'+case_folder
case_folder_1 = case_folder +'/'+result_folder
save_path_1 = program_path +'/'+case_folder_1
if not os.path.exists(save_path_1):
    os.makedirs(save_path_1)

log_name = 'logtest1.log'
data_folder = 'data2/'
data_path = program_path + '/'+ data_folder
test_loader=MeshAirfoilDataset(data_path,mode='test')

in_channels = test_loader.get(0).x.shape[-1]
out_channels = test_loader.get(0).y.shape[-1]

n_fig = 12
SEED_TEST = 8765
scale_mesh_test = [1.0,1.0,1.0]
zoom = True
xb=[-1.5,2.5]
yb=[-2.0,2.0]
set_normalized = True #False
modelname = 'GAT' #'GCN'
lambda_gnn = 0.8


##########################
# test
##########################
root_logger = logger_setup(os.path.join(save_path, log_name))
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
data=test_loader.get(0,mode=2)
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

root_logger.info("lambda_gnn = " + str(lambda_gnn))
root_logger.info("model name: "+modelname)
criterion =torch.nn.MSELoss().to(device)
state_dict = torch.load(save_path + '/model.pkl',map_location=device)
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict,"module.")
model.load_state_dict(state_dict)

root_logger.info("===========parameters============") 
root_logger.info("case name: "+case_folder) 
root_logger.info("result name: "+result_folder)
root_logger.info("proc_layers = "+str(proc_layers))
root_logger.info("hidden_channels = "+str(hidden_channels))
root_logger.info("num_layers = "+str(num_layers))
root_logger.info("latent_size = "+str(latent_size))
root_logger.info("scale_mesh_test= "+str(scale_mesh_test[0])+", "+str(scale_mesh_test[1])+", "+str(scale_mesh_test[2]))
root_logger.info("===========start test===========")
dataset_test=[]

with open(data_path + 'train_max_min.pkl', 'rb') as f:
    normalization_factors = pickle.load(f)
    data_max, data_min = normalization_factors
mode_data = 2

for i in range(test_loader.len):
    data=test_loader.get(i,mode=mode_data)
    #data = process_data(data,scale_mesh_test,mode=mode_data)
    if i<5:
        print("scale = ",scale_mesh_test,", x.shape = ",data.x.shape, ", edge_index.shape = ",data.edge_index.shape)
    dataset_test.append(data)

test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True, generator = torch.Generator().manual_seed(SEED_TEST))
model.eval()
sum_time=0
with torch.no_grad():
     sum_loss=0
     i_batch = 0
     for batch in test_loader:
         i_batch += 1
         batch=batch.to(device)
         truefield=batch.y
         start = time.perf_counter()
         prefield=model(batch)
         end = time.perf_counter()
         root_logger.info("time (ms): " + str((end - start) * 1000))
         sum_time += (end - start)
         if not set_normalized:
             for i in range(prefield.shape[1]):
                 truefield[:, i] = (truefield[:, i] + 1.0) / 2.0 * (data_max[i] - data_min[i]) + data_min[i]
                 prefield[:, i] = (prefield[:, i] + 1.0) / 2.0 * (data_max[i] - data_min[i]) + data_min[i]
         mes_loss=criterion(prefield,truefield)
         loss=mes_loss.cpu()
         sum_loss+=loss.item()
         if i_batch<=n_fig:
             batch = batch.cpu()
             truefield = truefield.cpu()
             prefield = prefield.cpu()
             log_images_1(batch.pos, prefield,truefield, batch.batch,batch.elems_list[0], 'test',file=case_folder_1+'/sample_'+str(i_batch)+'_',zoom=zoom,xb=xb,yb=yb)
             root_logger.info("batch i = " + str(i_batch))
             root_logger.info("spd = " + str(batch.spd.numpy()[0]))
             root_logger.info("aoa = " + str(batch.aoa.numpy()[0]))
             root_logger.info("loss =  " + str(loss.item()))
             #
             print("pos.shape = ", np.array(batch.pos).shape)
             print("prefield.shape = ", np.array(prefield).shape)
             
             field_data = {}
             field_data['pos'] = batch.pos.tolist()
             field_data['truefield'] = truefield.tolist()
             field_data['prefield'] = prefield.tolist()
             print("len = ",len(batch.elems_list[0]))
             field_data['elements'] = batch.elems_list[0]
             field_file = save_path_1 + \
                          '/data_spd_'+str(batch.spd.numpy()[0])+'_aoa_'+str(batch.aoa.numpy()[0])+'.pkl'
             with open(field_file,'wb') as f:
                 pickle.dump(field_data,f)

             print("truefield max = ", np.amax(np.array(truefield)))
             print("truefield min = ", np.amin(np.array(truefield)))

sum_loss1=sum_loss/(len(test_loader))
root_logger.info("        test_loss")
root_logger.info("        " + str(sum_loss1))
sum_time1=sum_time/(len(test_loader))
root_logger.info("        mean_time (s): "+ str(sum_time1))


