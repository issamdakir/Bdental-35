import bpy, os, sys
from os.path import join, dirname, exists, abspath
from .Operators.BDENTAL_Utils import *


ADDON_DIR = dirname(abspath(__file__))
Addon_Version_Path = join(ADDON_DIR, "Resources", "BDENTAL_Version.txt")
if exists(Addon_Version_Path):
    with open(Addon_Version_Path, "r") as rf:
        lines = rf.readlines()
        Addon_Version_Date = lines[0].split(";")[0]
else:
    Addon_Version_Date = "  "
# Selected icons :
red_icon = "COLORSET_01_VEC"
orange_icon = "COLORSET_02_VEC"
green_icon = "COLORSET_03_VEC"
blue_icon = "COLORSET_04_VEC"
violet_icon = "COLORSET_06_VEC"
yellow_icon = "COLORSET_09_VEC"
yellow_point = "KEYTYPE_KEYFRAME_VEC"
blue_point = "KEYTYPE_BREAKDOWN_VEC"

Wmin, Wmax = -1000, 10000


class BDENTAL_PT_MainPanel(bpy.types.Panel):
    """Main Panel"""

    bl_idname = "BDENTAL_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "BDENTAL"
    # bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props

        # Draw Addon UI :
        layout = self.layout

        box = layout.box()

        row = box.row()
        row.alert = True
        row.alignment = "EXPAND"
        row.label(text=f"VERSION : {Addon_Version_Date}")

        #######################
        
        split = box.split()
        col = split.column()
        col.operator("wm.open_mainfile", text="Open", icon="FILE_FOLDER")

        col = split.column()
        if bpy.data.is_dirty :
            col.alert = True
        col.operator("wm.save_mainfile", text="Save", icon="FOLDER_REDIRECT")

        col = split.column()
        col.operator("wm.save_as_mainfile", text="Save As...", icon="FILE_BLEND")
        
        split = box.split()
        col = split.column()
        col.operator("ed.undo", text="Undo", icon="LOOP_BACK")
        col = split.column()
        col.operator("ed.redo", text="Redo", icon="LOOP_FORWARDS")
        col = split.column()
        col.operator("wm.revert_mainfile", text="Revert", icon="FILE_REFRESH")

        box = layout.box()
        split = box.split(factor=1 / 3, align=False)
        col= split.column()
        col.label(text="Project Name :")
        col = split.column()
        col.prop(BDENTAL_Props, "ProjectNameProp", text="")

        box = layout.box()
        split = box.split(factor=1 / 3, align=False)
        col= split.column()
        col.label(text="Project Directory :")
        col = split.column()
        col.prop(BDENTAL_Props, "UserProjectDir", text="")

class BDENTAL_PT_DicomPanel(bpy.types.Panel):
    """Dicom Panel"""

    bl_idname = "BDENTAL_PT_DicomPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "DICOM"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):


        BDENTAL_Props = context.scene.BDENTAL_Props

        OrganizeInfoProp = BDENTAL_Props.OrganizeInfoProp
        layout = self.layout

        if BDENTAL_Props.UserProjectDir:
            box = layout.box()
            split = box.split(factor=1 / 3, align=False)
            col= split.column()
            col.label(text="DataType :")
            col= split.column()
            col.prop(BDENTAL_Props, "DataType", text="")

            if BDENTAL_Props.DataType == "DICOM Series":

                split = box.split(factor=1 / 3, align=False)
                col= split.column()
                col.label(text="DICOM Folder :")
                col= split.column()
                col.prop(BDENTAL_Props, "UserDcmDir", text="")
                
                if BDENTAL_Props.UserDcmDir:
                    # row = box.row()
                    # row.prop(BDENTAL_Props, "Dicom_Series_mode", expand=True)

                    if BDENTAL_Props.Dicom_Series_mode == "Advanced Mode":

                        if not BDENTAL_Props.Dicom_Series.startswith('Series'):

                            box = layout.box()
                            row = box.row()
                            row.operator(
                                    "wm.bdental_organize"
                                )
                        else :
                            split = box.split(factor=1 / 3, align=False)
                            col= split.column()
                            col.label(text="DICOM Series :")
                            col= split.column()
                            col.prop(BDENTAL_Props, "Dicom_Series", text="")

                            # serie = BDENTAL_Props.Dicom_Series
                            # infoDict = eval(OrganizeInfoProp)
                            # info = infoDict[serie]
                            # Count, Name, Date, Descript = info['Count'], info['Patient Name'], info['Series Date'], info['Series Description']

                            # layout.separator()
                            # row = layout.row()
                            # row.label(text=" DICOM Series Info :")

                            # box = layout.box()
                            # row = box.row()
                            # row.label(text=f"Files count : {Count}")
                            # row = box.row()
                            # row.label(text=f"Patient Name : {Name}")
                            # row = box.row()
                            # row.label(text=f"Series Date : {Date}")
                            # row = box.row()
                            # row.label(text=f"Series Description : {Descript}")

                            row = box.row()
                            row.prop(BDENTAL_Props, "scan_resolution", expand=True)

                            layout.separator()
                            box = layout.box()
                            row = box.row()
                            row.operator(
                                    "wm.bdental_volume_render"
                                )

                    if BDENTAL_Props.Dicom_Series_mode == "Simple Mode":
                        layout.separator()
                        box = layout.box()
                        row = box.row()
                        row.operator(
                                "wm.bdental_volume_render"
                            )

            if BDENTAL_Props.DataType == "3D Image File":

                split = box.split(factor=1 / 3, align=False)
                col= split.column()
                col.label(text="3D IMAGE File :")
                col= split.column()
                col.prop(BDENTAL_Props, "UserImageFile", text="")

                if BDENTAL_Props.UserImageFile:
                    layout.separator()
                    row = box.row()
                    row.prop(BDENTAL_Props, "scan_resolution", expand=True)
                    if not BDENTAL_Props.scan_resolution :
                        row = box.row()
                        row.operator(
                                "wm.bdental_organize"
                            )
                    else :
                        row = box.row()
                        row.operator(
                                "wm.bdental_volume_render"
                            )

        if context.object:
            N = context.object.name
            if "BD" in N and ("_CTVolume" in N or "_SLICES_POINTER" in N) :
                Preffix = N[0:6]
                DcmInfoDict = eval(BDENTAL_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]
                
                
                Soft,Bone,Teeth = -400, 700, 1400

                row = layout.row()
                row.label(text=f"Threshold {Wmin} to {Wmax} HU :")

                box = layout.box()
                row = box.row()
                row.prop(BDENTAL_Props, "TresholdMin", text="TRESHOLD Minimum", slider=True)
                row = box.row()
                row.prop(BDENTAL_Props, "TresholdMax", text="TRESHOLD Maximum", slider=True)

                layout.separator()

                row = layout.row()
                row.label(text="DICOM TO MESH :")

                Box = layout.box()

                Box = layout.box()
                row = Box.row()
                row.prop(BDENTAL_Props, "SoftTreshold", text="Soft Tissu")
                row.prop(BDENTAL_Props, "SoftSegmentColor", text="")
                row.prop(BDENTAL_Props, "SoftBool", text="")
                row = Box.row()
                row.prop(BDENTAL_Props, "BoneTreshold", text="Bone")
                row.prop(BDENTAL_Props, "BoneSegmentColor", text="")
                row.prop(BDENTAL_Props, "BoneBool", text="")

                row = Box.row()
                row.prop(BDENTAL_Props, "TeethTreshold", text="Teeth")
                row.prop(BDENTAL_Props, "TeethSegmentColor", text="")
                row.prop(BDENTAL_Props, "TeethBool", text="")

                Box = layout.box()
                row = Box.row()
                row.operator("wm.bdental_multitresh_segment")
              
class BDENTAL_PT_SlicesPanel(bpy.types.Panel):
    """Slices Panel"""

    bl_idname = "BDENTAL_PT_SlicesPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "SLICES"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.operator("wm.bdental_addslices", icon="EMPTY_AXIS")
                

        row = box.row()
        row.alert = True 
        mats = [mat for mat in bpy.data.materials if "_SLICE_mat" in mat.name]
        if not mats:
            row.alert = False 
                 
        row.prop(BDENTAL_Props, "slices_brightness", text="Brightness")
        row.prop(BDENTAL_Props, "slices_contrast", text="Contrast")
                
        row = box.row()
        row.label(text="Axial Slice Flip :")

        row = box.row()
        row.operator("wm.bdental_flip_camera_axial_90_plus", icon="PLUS")
        row.operator("wm.bdental_flip_camera_axial_90_minus", icon="REMOVE")
        row.operator("wm.bdental_flip_camera_axial_up_down", icon="TRIA_UP")
        row.operator("wm.bdental_flip_camera_axial_left_right", icon="TRIA_RIGHT")

        row = box.row()
        row.label(text="Coronal Slice Flip :")

        row = box.row()
        row.operator("wm.bdental_flip_camera_coronal_90_plus", icon="PLUS")
        row.operator("wm.bdental_flip_camera_coronal_90_minus", icon="REMOVE")
        row.operator("wm.bdental_flip_camera_coronal_up_down", icon="TRIA_UP")
        row.operator("wm.bdental_flip_camera_coronal_left_right", icon="TRIA_RIGHT")

        row = box.row()
        row.label(text="Sagittal Slice Flip :")

        row = box.row()
        row.operator("wm.bdental_flip_camera_sagittal_90_plus", icon="PLUS")
        row.operator("wm.bdental_flip_camera_sagittal_90_minus", icon="REMOVE")
        row.operator("wm.bdental_flip_camera_sagittal_up_down", icon="TRIA_UP")
        row.operator("wm.bdental_flip_camera_sagittal_left_right", icon="TRIA_RIGHT")

class BDENTAL_PT_ToolsPanel(bpy.types.Panel):
    """Tools Panel"""

    bl_idname = "BDENTAL_PT_ToolsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "TOOLS"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout

        #import / export :
        layout.label(text="IMPORT/EXPORT :", icon=yellow_point)
        Box = layout.box()
        row = Box.row()
        row.operator("wm.bdental_import_mesh", icon="IMPORT")
        row.operator("wm.bdental_export_mesh", icon="EXPORT")

        layout.separator()
        
        # Model color :
        layout.label(text="COLOR :", icon=yellow_point)
        Box = layout.box()
        row = Box.row()
        row.operator("wm.bdental_add_color", text="Add Color", icon="MATERIAL")
        if context.object:
            mat = context.object.active_material
            if mat:
                row.prop(mat, "diffuse_color", text="")
            else:
                row.prop(BDENTAL_Props, "no_material_prop", text="")

        row.operator("wm.bdental_remove_color", text="Remove Color")

        # Join / Link ops :
        layout.separator()
        layout.label(text="Objects Relations :", icon=yellow_point)

        Box = layout.box()
        row = Box.row()
        row.operator("wm.bdental_parent_object", text="Parent", icon="LINKED")
        row.operator("wm.bdental_join_objects", text="Join", icon="SNAP_FACE")
        row.operator("wm.bdental_lock_objects", text="Lock", icon="LOCKED")

        row = Box.row()
        row.operator("wm.bdental_unparent_objects", text="Un-Parent", icon="LIBRARY_DATA_OVERRIDE")
        row.operator("wm.bdental_separate_objects", text="Separate", icon="SNAP_VERTEX")
        row.operator("wm.bdental_unlock_objects", text="Un-Lock", icon="UNLOCKED")

        # Model Repair Tools :
        layout.separator()
        layout.label(text="REPAIR TOOLS", icon=yellow_point)
        Box = layout.box()
        
        split = Box.split(factor=2 / 3, align=False)
        col = split.column()
        row = col.row(align=True)
        row.operator("wm.bdental_decimate", text="Decimate", icon="MOD_DECIM")
        row.prop(BDENTAL_Props, "decimate_ratio", text="")
        row = col.row()
        row.operator("wm.bdental_fill", text="Fill", icon="OUTLINER_OB_LIGHTPROBE")
        row.operator(
            "wm.bdental_retopo_smooth", text="Retopo-Smooth", icon="SMOOTHCURVE")
        try:
            ActiveObject = bpy.context.view_layer.objects.active
            if ActiveObject:
                if ActiveObject.mode == "SCULPT":
                    row.operator(
                        "sculpt.sample_detail_size", text="", icon="EYEDROPPER"
                    )
        except Exception:
            pass

        col = split.column()
        
        row = col.row()
        # row.scale_y = 2
        row.operator("wm.bdental_clean_mesh2", text="Clean Mesh", icon="BRUSH_DATA")
        row = col.row()
        row.operator("wm.bdental_voxelremesh", text="Remesh", icon="MOD_REMESH")
        
        row = Box.row()
        row.operator("wm.bdental_normals_toggle")
        row.operator("wm.bdental_flip_normals")

        # Cutting Tools :
        layout.row().separator()
        layout.label(text="Cutting Tools :", icon=yellow_point)
        Box = layout.box()
        row = Box.row()
        row.prop(BDENTAL_Props, "Cutting_Tools_Types_Prop", text="")

        if BDENTAL_Props.Cutting_Tools_Types_Prop == "Curve Cutter 1":
            row = Box.row()
            row.prop(BDENTAL_Props, "CurveCutCloseMode", text="")
            row.operator(
                "wm.bdental_curvecutteradd", text="Add Cutter", icon="GP_SELECT_STROKES"
            )
            row = Box.row()
            row.operator(
                "wm.bdental_curvecuttercut", text="Perform Cut", icon="GP_MULTIFRAME_EDITING"
            )
        

        elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Curve Cutter 2":
            row = Box.row()
            row.prop(BDENTAL_Props, "CurveCutCloseMode", text="")
            row.operator(
                "wm.bdental_curvecutteradd2", text="Add Cutter", icon="GP_SELECT_STROKES"
            )
            row = Box.row()
            row.operator(
                "wm.bdental_curvecutter2_shortpath",
                text="Perform Cut",
                icon="GP_MULTIFRAME_EDITING",
            )
        # elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Curve Cutter 3":
        #     row = Box.row()
        #     row.prop(BDENTAL_Props, "CurveCutCloseMode", text="")
        #     row.operator(
        #         "wm.bdental_curvecutteradd3", text="ADD CUTTER", icon="GP_SELECT_STROKES"
        #     )
        #     row.operator(
        #         "wm.bdental_curvecuttercut3",
        #         text="CUT",
        #         icon="GP_MULTIFRAME_EDITING",
        #     )
        elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Square Cutter":

            # Cutting mode column :
            row = Box.row()
            row.label(text="Select Cutting Mode :")
            row.prop(BDENTAL_Props, "cutting_mode", text="")

            row = Box.row()
            row.operator("wm.bdental_add_square_cutter", text="Add Square Cutter")
            # row.operator("wm.bdental_square_cut_confirm", text="Perform Cut")
            # row.operator("wm.bdental_square_cut_exit", text="EXIT")

        elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Paint Cutter":

            row = Box.row()
            row.operator("wm.bdental_paintarea_toggle", text="Add Cutter")
            row.operator("wm.bdental_paintarea_plus", text="", icon="ADD")
            row.operator("wm.bdental_paintarea_minus", text="", icon="REMOVE")
            row = Box.row()
            row.operator("wm.bdental_paint_cut", text="Perform Cut")
        
        elif BDENTAL_Props.Cutting_Tools_Types_Prop == "Path Cutter":
            row = Box.row()
            row.operator("wm.bdental_add_path_cutter")

        if context.active_object:
            if (
                "BDENTAL_Curve_Cut" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):

                obj = context.active_object
                row = Box.row()
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        # Make BaseModel, survey, Blockout :
        layout.separator()
        layout.label(
            text="Dental Model Tools :", icon=yellow_point
        )
        Box = layout.box()
        row = Box.row()
        row.alignment = "CENTER"
        row.prop(BDENTAL_Props, "BaseHeight")
        row.operator("wm.bdental_model_base", text="Make Model Base")
        row.operator("wm.bdental_add_offset", text="Add Offset")
        row = Box.row()
        row.alignment = "CENTER"
        row.operator("wm.bdental_undercuts_preview", text="Preview Undercuts")
        row.operator("wm.bdental_blockout_new", text="Blocked Model")
        
        Box = layout.box()
        Box.label(text="Teeth :")
        row = Box.row()
        row.alignment = "CENTER"
        row.prop(BDENTAL_Props, "TeethLibrary", text="")
        row.operator("wm.bdental_add_teeth")

        Box = layout.box()
        row = Box.row()
        row.prop(BDENTAL_Props, "text", text="3D Text")
        row = Box.row()
        row.operator("wm.bdental_add_3d_text", text="Add 3D Text")

        # row = Box.row()
        # row.operator("wm.bdental_overhangs_preview")

class BDENTAL_PT_ImplantPanel(bpy.types.Panel):
    """ Implant panel"""

    bl_idname = "BDENTAL_PT_ImplantPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "IMPLANT"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout

        Box = layout.box()
        Box.alignment = "EXPAND"
        Box.label(text="Implant :")
        row = Box.row()
        row.operator("wm.bdental_slices_pointer_select")
        
        row = Box.row()
        row.operator("wm.bdental_add_implant")
        row.operator("wm.bdental_align_implants")

        row = Box.row()
        row.operator("wm.bdental_lock_to_pointer")
        row.operator("wm.bdental_unlock_from_pointer")

        row = Box.row()
        row.operator("wm.bdental_implant_to_pointer")
        row.operator("wm.bdental_pointer_to_implant")

        row = Box.row()
        row.label(text="Pointer Jump :")
        row.operator("wm.bdental_fly_previous", text="", icon="TRIA_LEFT")
        row.operator("wm.bdental_fly_next", text="", icon="TRIA_RIGHT")
        row.operator("wm.bdental_remove_info_footer", icon="CANCEL")
        

        # layout.separator()

        # Box = layout.box()
        # Box.label(text="Sleeve/Pin :")
        # row = Box.row()
        # row.alignment = "CENTER"
        # row.prop(BDENTAL_Props, "SleeveDiameter")
        # row.prop(BDENTAL_Props, "SleeveHeight")
        # row = Box.row()
        # row.alignment = "CENTER"
        # row.prop(BDENTAL_Props, "HoleDiameter")
        # row.prop(BDENTAL_Props, "HoleOffset")

        # row = Box.row()
        # row.alignment = "CENTER"
        # row.operator("wm.bdental_add_implant_sleeve")

class BDENTAL_PT_Guide(bpy.types.Panel):
    """ Guide Panel"""

    bl_idname = "BDENTAL_PT_Guide"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "SURGICAL GUIDE"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        layout = self.layout

        Box = layout.box()
        

        row = Box.row()
        # row.operator("wm.bdental_add_guide_splint")
        row.operator("wm.bdental_add_guide_splint_geom")

        row = Box.row()
        row.operator("wm.bdental_add_tube")
        row.prop(BDENTAL_Props, "TubeCloseMode", text="")

        if context.active_object:
            if (
                "BDENTAL_GuideTube" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):
                obj = context.active_object
                row = Box.row()
                row.prop(obj.data, "bevel_depth", text="Radius")
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        # row = Box.row()
        # row.operator("wm.bdental_add_guide_cutters_from_sleeves")

        row = Box.row()
        row.operator("wm.bdental_guide_add_component")

        # row = Box.row()
        # row.operator("wm.bdental_set_guide_components", icon="PLUS")
        # row = Box.row()
        # row.operator("wm.bdental_guide_set_cutters", icon="REMOVE")
        

        
        
        row = Box.row()
        row.operator("wm.bdental_guide_finalise")

        row = Box.row()
        row.operator(
            "wm.bdental_splintcutteradd",
            text="ADD SPLINT CUTTER",
            icon="GP_SELECT_STROKES",
        )
        row.prop(BDENTAL_Props, "CurveCutCloseMode", text="")
        if context.active_object:
            if (
                "BDENTAL_Splint_Cut" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):
                obj = context.active_object
                row = Box.row()
                row.prop(obj.data, "bevel_depth", text="Radius")
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        row = Box.row()
        row.operator(
            "wm.bdental_splintcuttercut", text="CUT", icon="GP_MULTIFRAME_EDITING"
        )

####################################################################
class BDENTAL_PT_Align(bpy.types.Panel):
    """ALIGN Panel"""

    bl_idname = "BDENTAL_PT_Main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL"
    bl_label = "ALIGN"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_Props = context.scene.BDENTAL_Props
        AlignModalState = BDENTAL_Props.AlignModalState
        layout = self.layout

        # Align Tools :
        layout.separator()
        Box = layout.box()
        row = Box.row()
        row.label(text="Align Tools")

        row = Box.row()
        row.operator("wm.bdental_align_to_front", text="Align To Me", icon="AXIS_FRONT")

        row = Box.row()
        row.operator("wm.bdental_to_center", text="Move To Center", icon="SNAP_FACE_CENTER")
        
        row = Box.row()
        row.operator("wm.bdental_align_to_active", text="Align To Active")
        
        # row.operator("wm.bdental_align_to_cursor", text="Move To Cursor", icon="AXIS_FRONT")
        
        
        split = Box.split(factor=2 / 3, align=False)
        col = split.column()
        row = col.row()
        row.operator("wm.bdental_occlusalplane", text="OCCLUSAL PLANE")
        col = split.column()
        row = col.row()
        row.alert = True
        row.operator("wm.bdental_occlusalplaneinfo", text="INFO", icon="INFO")

        #Auto align :
        # box = layout.box()
        # row = box.row()
        # row.operator("wm.bdental_auto_align_icp", text="AUTO ALIGN")

        # Align Points and ICP :
        split = layout.split(factor=2 / 3, align=False)
        col = split.column()
        row = col.row()
        row.operator("wm.bdental_alignpoints", text="ALIGN")
        col = split.column()
        row = col.row()
        row.alert = True
        row.operator("wm.bdental_alignpointsinfo", text="INFO", icon="INFO")

        Condition_1 = len(bpy.context.selected_objects) != 2
        Condition_2 = bpy.context.selected_objects and not bpy.context.active_object
        Condition_3 = bpy.context.selected_objects and not (
            bpy.context.active_object in bpy.context.selected_objects
        )
        Condition_4 = not bpy.context.active_object in bpy.context.visible_objects

        Conditions = Condition_1 or Condition_2 or Condition_3 or Condition_4
        if AlignModalState:
            self.AlignLabels = "MODAL"
        else:
            if Conditions:
                self.AlignLabels = "INVALID"

            else:
                self.AlignLabels = "READY"

        #########################################

        if self.AlignLabels == "READY":
            TargetObjectName = context.active_object.name
            SourceObjectName = [
                obj
                for obj in bpy.context.selected_objects
                if not obj is bpy.context.active_object
            ][0].name

            box = layout.box()

            row = box.row()
            row.alert = True
            row.alignment = "EXPAND"
            row.label(text="READY FOR ALIGNEMENT.")

            row = box.row()
            row.alignment = "EXPAND"
            row.label(text=f"{SourceObjectName} will be aligned to, {TargetObjectName}")

        if self.AlignLabels == "INVALID" or self.AlignLabels == "NOTREADY":
            box = layout.box()
            row = box.row(align=True)
            row.alert = True
            row.alignment = "EXPAND"
            row.label(text="STANDBY MODE", icon="ERROR")

        if self.AlignLabels == "MODAL":
            box = layout.box()
            row = box.row()
            row.alert = True
            row.alignment = "EXPAND"
            row.label(text="WAITING FOR ALIGNEMENT...")

##########################################################################################
# Registration :
##########################################################################################

classes = [
    BDENTAL_PT_MainPanel,
    BDENTAL_PT_DicomPanel,
    BDENTAL_PT_SlicesPanel,
    BDENTAL_PT_ImplantPanel,
    BDENTAL_PT_Guide,
    BDENTAL_PT_Align,
    BDENTAL_PT_ToolsPanel,
    
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

