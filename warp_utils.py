from warp.render import render_usd
import numpy as np

class NewRenderer(render_usd.UsdRenderer):
    def __init__(self, stage, up_axis="y", fps=60, scaling=1):
        super().__init__(stage, up_axis, fps, scaling)
        # self.up_axis = up_axis.upper()

    def render_capsules(self, name: str, points, radius, half_height, colors=None, orientations=None):
        from pxr import UsdGeom, Gf, Sdf
        up_axis = self.up_axis.upper()

        instancer_path = self.root.GetPath().AppendChild(name)
        instancer = UsdGeom.PointInstancer.Get(self.stage, instancer_path)
        
        radius_is_scalar = np.isscalar(radius)
        if not instancer:
            if colors is None:
                instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
                
                # Create a capsule instead of a sphere
                instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
                instancer_capsule.GetAxisAttr().Set(up_axis)
                
                if radius_is_scalar:
                    instancer_capsule.GetRadiusAttr().Set(radius)
                    instancer_capsule.GetHeightAttr().Set(2.0 * half_height)
                else:
                    # Assuming half_heights is a similar array like radius for individual heights
                    # If not, adjust accordingly
                    # instancer_capsule.GetRadiusAttr().Set(radius)
                    instancer_capsule.GetRadiusAttr().Set(1.0)
                    instancer_capsule.GetHeightAttr().Set(2.0 * half_height)
                    # Set scales for the PointInstancer to adjust the size of the instances
                    # scales = np.column_stack((np.ones(len(radius)), np.ones(len(radius)), np.ones(len(radius))))
                    if up_axis == 'Y': scales = np.column_stack((radius*.6, [2.0*half_height]*len(radius), radius ))
                    elif up_axis =='Z': scales = np.column_stack((radius*.6, radius, [2.0*half_height]*len(radius)))
                    # scales = np.column_stack((radius, [2.0*half_height]*len(radius), radius))
                    instancer.GetScalesAttr().Set(scales)
                instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
                instancer.CreateProtoIndicesAttr().Set([0] * len(points))

                # Set identity rotations
                quats = [Gf.Quath(0.0, 0.0, 0.0, 1.0)] * len(points)
                instancer.GetOrientationsAttr().Set(quats, self.time)
            else:
                instancer = UsdGeom.Points.Define(self.stage, instancer_path)
                instancer.GetAxisAttr().Set(up_axis)
                instancer.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Float3Array, "vertex", 1)
                        
        if orientations is not None and len(orientations) > 0:
            instancer.GetOrientationsAttr().Set(orientations, self.time)
        else:
            # Set identity rotations if orientations are not provided
            quats = [Gf.Quath(0.0, 0.0, 0.0, 1.0)] * len(points)
            instancer.GetOrientationsAttr().Set(quats, self.time)
            
        if colors is None:
            instancer.GetPositionsAttr().Set(points, self.time)
        else:
            instancer.GetPointsAttr().Set(points, self.time)
            instancer.GetDisplayColorAttr().Set(colors, self.time)