import os
import pyvista as pv
# case文件夹路径
case_directory = r'E:\naca\TrainDateset'
# 输出文件夹
output_directory = r'E:\naca\TrainDateset\out'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(case_directory):
    if filename.endswith('.case'):
        case_file_path=os.path.join(case_directory, filename)
        # 读取后遍历块
        multiblock = pv.read(case_file_path)
        combined_mesh=None
        for i in range(len(multiblock)):
            mesh=multiblock[i]
            # 物理量搜索后输出vtu，按照需要修改
            if mesh is not None and 'Pressure' in mesh.array_names:
                print(f"Data found in block_{i}.")
                if mesh.cell_data and not mesh.point_data:
                    print(f"Converting cell data to point data for block {i} in {filename}.")
                    mesh=mesh.cell_data_to_point_data()
                block_name=multiblock.get_block_name(i)
                if not block_name:
                    block_name=f"block_{i}"
                block_name=block_name.strip().replace(" ","_")
                # 块累加
                if combined_mesh is None:
                    combined_mesh=mesh
                else:
                    combined_mesh=combined_mesh + mesh
        vtu_filename = f"{os.path.splitext(filename)[0]}.vtu"
        vtu_file_path = os.path.join(output_directory, vtu_filename)
        if combined_mesh is not None:
            combined_mesh.save(vtu_file_path, binary=False)
            print(f"Saved combined data to {vtu_file_path} successfully.")
        else:
            print(f"No valid data found in {filename}.")
print("All conversions are complete.")


