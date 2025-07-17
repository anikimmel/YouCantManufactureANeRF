from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.gp import gp_Pnt, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
import sys
import os
import gc


def create_mesh(input_step_file, output_mesh_file):
    # 1) Load your STEP
    reader = STEPControl_Reader()
    reader.ReadFile(input_step_file)
    reader.TransferRoots()

    # 3) Extract the shape—OneShape() will return a compound if there
    #    were multiple roots, or the single shape otherwise.
    shape = reader.OneShape()
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0, 0, 0), 0.1)
    shape = BRepBuilderAPI_Transform(shape, trsf).Shape()

    write_stl_file(
        a_shape=shape,
        filename=output_mesh_file,
        mode="binary",
        linear_deflection=0.025,  # e.g. 0.05 mm max distance
        angular_deflection=0.025  # e.g. 0.05 rad max angle
    )

    print(f"Imported {input_step_file}\n Exported high-precision mesh to {output_mesh_file}")


if __name__ == '__main__':
    input_dir = '/path/to/input_dir/'
    output_dir = '/path/to/output_dir/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cnt = 0
    for file in os.listdir(input_dir):
        if file.endswith(".stp"):
            try:
                print(f'Count: {cnt}')
                step_filename = os.path.join(input_dir, file)
                output_filename = os.path.join(output_dir, file.replace('.stp', '.stl'))
                print(step_filename)
                print(output_filename)
                if not os.path.exists(output_filename):
                    create_mesh(step_filename, output_filename)
                    cnt += 1
                else:
                    print(f"File {output_filename} already exists, skipping")
            except Exception as e:
                print(str(e))
                print(file)
