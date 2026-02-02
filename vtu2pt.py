#
import numpy as np
import pyvista as pv
import os
import pickle

def vtp_to_graph(vtp_file, pkl_file, max_min):
    mesh = pv.read(vtp_file)
    points = np.array(mesh.points, dtype = np.float32)
    elements = np.array(mesh.cells, dtype = np.int64)
    celltypes = mesh.celltypes

    print("len of celltypes: ",len(celltypes))
    print(celltypes)

    if 'Pressure' in mesh.point_data: #cell_data
        print("Reading pressure from data")
        fields = np.array(mesh.point_data['Pressure'], dtype = np.float32).reshape(-1,1) #cell_data
    else:
        print("Error: Pressure not found")
        fields = points[:,2].reshape(-1,1)

    edges = []
    elems = []
    i = 0
    while i<elements.shape[0]:
        n = elements[i]
        current_element = elements[i+1:i+n+1]
        edges += [[current_element[i], current_element[(i + 1) % n]] for i in range(n)]
        elems += [current_element.tolist()]
        i = i+n+1

    #data = Data(x=pr_data, pos=points, edge_index=edges, faces=faces)
    data = {}
    data['pos'] = points
    data['elems'] = elems
    data['edges'] = np.array(edges, dtype = np.int64)
    data['fields'] = fields

    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

    print("pos shape: ",data['pos'].shape)
    print("edges shape: ",data['edges'].shape)
    print("fields shape: ",data['fields'].shape)
    print("elems len: ",len(data['elems']))

    fields_max, fields_min = max_min
    p_max = np.amax(fields)
    p_min = np.amin(fields)
    if fields_max==None or fields_min==None:
        fields_max = p_max
        fields_min = p_min
    else:
        if p_max>fields_max:
            fields_max = p_max
        if p_min<fields_min:
            fields_min = p_min
    max_min = [fields_max, fields_min]
    return max_min

if __name__ == "__main__":
    input_dir = 'D:/project/sample_0012_0015/' #'D:/project/aoa0-10_v0.5-5/'
    output_dir = 'D:/project/sample_0012_0015_processed/' #'D:/project/aoa0-10_v0.5-5_processed/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    max_min = [None, None]
    files = os.listdir(input_dir)
    for input_file in files:
        if input_file.rsplit('.', 1)[1]=='vtu': #'vtp'
            output_file = input_file.rsplit('.', 1)[0]+'.pkl'
            max_min = vtp_to_graph(input_dir+input_file, output_dir+output_file,max_min)

    print("max_min: ", max_min)
    max_min = np.array([[max_min[0]], [max_min[1]]])
    with open(output_dir+'train_max_min.pkl', 'wb') as f:
        pickle.dump(max_min, f)