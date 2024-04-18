from pxr import Usd, UsdGeom

# Create a new USD stage with a root layer and a default prim
stage = Usd.Stage.CreateNew('simpleStage.usda')
stage.SetDefaultPrim(stage.DefinePrim('/SimpleStage', 'Xform'))

# Define a new Cube under the default prim
cube = UsdGeom.Cube.Define(stage, '/SimpleStage/Cube')

# Define an instance of the Cube
instance = UsdGeom.Xform.Define(stage, '/SimpleStage/Instance')
instance_instance = UsdGeom.Xform.Define(stage, '/SimpleStage/Instance/InstanceCube')

# Add reference to the Cube prim
instance_instance.GetPrim().GetReferences().AddReference('simpleStage.usda', cube.GetPrim().GetPath())

# Traverse the scene graph and print the paths of prims, including instance proxies
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    print(prim.GetPath())

# Save the stage to a USD file
stage.Save()
