
try:
    import bpy, bgl, blf, mathutils
except ImportError:
    raise ImportError("Failed to import Blender modules. This script can't run "
    "standalone. Run `blender --python scripts/blender_make_composite_glb.py` from the "
    "gala_kinematic root directory. Watch the terminal for "
    "debug/error output. You can visually verify the resulting composite glb in the "
    "Blender GUI window that pops up, or just close the window.")

from bpy import data, ops, props, types, context, utils

from mathutils import *; from math import *
import os
import glob


class ImportItem:
    def __init__(self, filepath, name=None, do_join_all=True, vertex_merge_distance=0.0, decimate_fraction=1.0, force_color=None,
        do_generate_gfx_replay=None, rotation=None):
        self.filepath = filepath
        # the do_join_all option is good for ReplicaCAD baked stages, for which all
        # nodes/meshes share the same (baked) material. (Joining to a single mesh won't 
        # work well if the items are using different materials.)
        self.do_join_all = do_join_all
        self.vertex_merge_distance = vertex_merge_distance
        self.decimate_fraction = decimate_fraction
        self.name = name
        self.force_color = force_color
        self.do_generate_gfx_replay = do_generate_gfx_replay
        self.rotation = rotation

source_data_path = "source_data/"
google_scanned_path = source_data_path + "google_scanned/"
ycb_dataset_path = source_data_path + "objects/ycb/"
replicacad_dataset_path = source_data_path + "replica_cad_baked_lighting/"
replicacad_stages_path = replicacad_dataset_path + "stages_uncompressed/"
replicacad_articulated_objects_path = replicacad_dataset_path + "urdf_uncompressed/"
fixed_replicacad_models_path = source_data_path + "fixed_replicacad_models/"
spot_path = source_data_path + "spot_arm_textured/spot_arm/spot_arm/meshes/"
stretch_path = source_data_path + "hab_stretch/meshes/"
preprocessed_fetch_meshes = source_data_path + "preprocessed_fetch_meshes/"
debug_models_path = source_data_path + "debug_models/"
mp3d_example_path = source_data_path + "scene_datasets/mp3d_example/"

# Core Galactic 3D models used for our paper.
include_fetch = True
include_ycb_objects = True
include_replicacad_stages = True
include_replicacad_articulated_objects = True
include_debug_models = True

# Additional 3D models that you may use when extending Galactic. See DATA.md.
include_spot = False
include_stretch = False
include_mp3d_example = False
include_google_scanned = False

output_path = "temp/"
output_composite_glb_filename = "composite.glb"

def main():

    import_items = []

    warnings = []
    def add_warning(warning_str):
        warnings.append(warning_str)

    def check_data_path(path, path_name):
        if not os.path.exists(path):
            raise FileNotFoundError("{} {} not found. See SOURCE_DATA.md "
            "for download instructions.".format(path_name, path))

    if not os.path.exists(source_data_path):
        raise FileNotFoundError("source_data_path {} not found. See DATA.md and be sure to run this script from the gala_kinematic root directory.".format(source_data_path))

    if include_spot:
        check_data_path(spot_path, "spot_path")
        import_items.extend([
            ImportItem(spot_path + "arm0.link_el0.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "arm0.link_el1.obj", force_color=[1, 1, 0]),
            ImportItem(spot_path + "arm0.link_fngr.obj", force_color=[0.5, 0.5, 0.5]),
            ImportItem(spot_path + "arm0.link_hr0.obj", force_color=[1, 1, 0]),
            ImportItem(spot_path + "arm0.link_sh0.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "arm0.link_sh1.obj", force_color=[0.5, 0.5, 0.5]),
            ImportItem(spot_path + "arm0.link_wr0.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "arm0.link_wr1.obj", force_color=[1, 1, 0]),
            ImportItem(spot_path + "base.obj", force_color=[0.3, 0.78, 1.0]),
            ImportItem(spot_path + "fl.hip.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "fl.lleg.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "fl.uleg.obj", force_color=[1, 1, 0]), # [0.3, 0.78, 1.0]),
            ImportItem(spot_path + "fr.hip.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "fr.lleg.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "fr.uleg.obj", force_color=[1, 1, 0]), # [0.3, 0.78, 1.0]),
            ImportItem(spot_path + "hl.hip.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "hl.lleg.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "hl.uleg.obj", force_color=[1, 1, 0]), # [0.3, 0.78, 1.0]),
            ImportItem(spot_path + "hr.hip.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "hr.lleg.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(spot_path + "hr.uleg.obj", force_color=[1, 1, 0]) # [0.3, 0.78, 1.0])
        ])

    if include_stretch:
        check_data_path(stretch_path, "stretch_path")
        # The override colors here aren't accurate to public Stretch images; they
        # are chosen to help visually distinguish different parts.
        import_items.extend([
            ImportItem(stretch_path + "link_head_tilt.obj", force_color=None),
            ImportItem(stretch_path + "link_lift.obj", force_color=[0.25, 0.25, 0.25]),
            ImportItem(stretch_path + "link_left_wheel.obj", decimate_fraction=0.1, force_color=None),
            ImportItem(stretch_path + "link_gripper_fingertip_right.obj", force_color=None),
            ImportItem(stretch_path + "link_respeaker.obj", decimate_fraction=0.1, force_color=None),
            ImportItem(stretch_path + "laser.obj", force_color=None),
            ImportItem(stretch_path + "link_mast.obj", force_color=[0.0, 0.0, 0.0]),
            ImportItem(stretch_path + "link_gripper_finger_left_gradient.glb", force_color=None),
            ImportItem(stretch_path + "link_gripper_finger_right_gradient.glb", force_color=None),
            ImportItem(stretch_path + "link_wrist_yaw.obj", force_color=None),
            ImportItem(stretch_path + "link_right_wheel.obj", decimate_fraction=0.1, force_color=None),
            ImportItem(stretch_path + "link_head_pan.obj", force_color=[0.1, 0.1, 0.1]),
            ImportItem(stretch_path + "omni_wheel_m.obj", decimate_fraction=0.1, force_color=None),
            ImportItem(stretch_path + "base_link.obj", force_color=None),
            ImportItem(stretch_path + "link_head.obj", force_color=[0.175, 0.175, 0.175]),
            ImportItem(stretch_path + "link_gripper_fingertip_left.obj", force_color=None),
            ImportItem(stretch_path + "link_arm_l4.obj", force_color=[0.0, 0.0, 0.0]),
            ImportItem(stretch_path + "link_arm_l3.obj", force_color=[0.15, 0.15, 0.15]),
            ImportItem(stretch_path + "link_arm_l2.obj", force_color=[0.175, 0.175, 0.175]),
            ImportItem(stretch_path + "link_arm_l1.obj", force_color=[0.2, 0.2, 0.2]),
            ImportItem(stretch_path + "link_arm_l0.obj", force_color=[0.225, 0.225, 0.225]),
            ImportItem(stretch_path + "link_aruco_inner_wrist.STL", force_color=None),
            ImportItem(stretch_path + "link_aruco_left_base.STL", force_color=None),
            ImportItem(stretch_path + "link_aruco_right_base.STL", force_color=None),
            ImportItem(stretch_path + "link_aruco_shoulder.STL", force_color=None),
            ImportItem(stretch_path + "link_aruco_top_wrist.STL", force_color=None),
            ImportItem(stretch_path + "link_straight_gripper.STL", decimate_fraction=0.1, force_color=[0.175, 0.175, 0.175]),
            ImportItem(stretch_path + "link_wrist_pitch.STL", decimate_fraction=0.1, force_color=[0.1, 0.1, 0.1]),
            ImportItem(stretch_path + "link_wrist_roll.STL", force_color=[0.1, 0.1, 0.1]),
            ImportItem(stretch_path + "link_wrist_yaw_bottom.STL", force_color=[0.2, 0.2, 0.2]),
            ImportItem(stretch_path + "/realsense2/d435.dae", decimate_fraction=0.04, force_color=None),
        ])        


    if include_replicacad_articulated_objects:
        check_data_path(replicacad_articulated_objects_path, "replicacad_articulated_objects_path")
        check_data_path(fixed_replicacad_models_path, "fixed_replicacad_models_path")
        import_items.extend([
            ImportItem(fixed_replicacad_models_path + "fridge/body_brighter2.gltf", vertex_merge_distance=0.01),
            ImportItem(fixed_replicacad_models_path + "fridge/bottom_door_brighter2.gltf", vertex_merge_distance=0.01),
            ImportItem(fixed_replicacad_models_path + "fridge/top_door_brighter2.gltf", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_counter/kitchen_counter.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_counter/drawer1.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_counter/drawer2.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_counter/drawer3.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_counter/drawer4.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_cupboards/kitchencupboard_base.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_cupboards/kitchencupboard_doorWhole_L.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_cupboards/kitchencupboard_doorWhole_R.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_cupboards/kitchencupboard_doorWindow_L.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "kitchen_cupboards/kitchencupboard_doorWindow_R.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "doors/door2.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "cabinet/cabinet.glb", vertex_merge_distance=0.01),
            ImportItem(replicacad_articulated_objects_path + "cabinet/door.glb", vertex_merge_distance=0.01),
            ImportItem(fixed_replicacad_models_path + "chest_of_drawers/chestOfDrawers_base.gltf", vertex_merge_distance=0.01),
            ImportItem(fixed_replicacad_models_path + "chest_of_drawers/chestOfDrawers_DrawerBot.gltf", vertex_merge_distance=0.01),
            ImportItem(fixed_replicacad_models_path + "chest_of_drawers/chestOfDrawers_DrawerMid.gltf", vertex_merge_distance=0.01),
            ImportItem(fixed_replicacad_models_path + "chest_of_drawers/chestOfDrawers_DrawerTop.gltf", vertex_merge_distance=0.01),
        ])


    if include_fetch:
        check_data_path(preprocessed_fetch_meshes, "preprocessed_fetch_meshes")
        import_items.extend([
            # this GLB is essentially already in the format we want, so we don't join or simplify
            ImportItem(preprocessed_fetch_meshes 
#                + "fetch_meshes_all_6k_verts_renamed_texdownsampled_smoothed.glb", do_join_all=False)
                + "fetch_meshes_all_6k_verts_renamed_texdownsampled_smoothed_recolored.glb", do_join_all=False)
        ])

    if include_debug_models:
        check_data_path(debug_models_path, "debug_models_path")
        import_items.extend([
            ImportItem(debug_models_path + "sphere_green_wireframe.glb"),
            ImportItem(debug_models_path + "sphere_orange_wireframe.glb", force_color=[1.0, 0.6, 0.0]),
            ImportItem(debug_models_path + "sphere_blue_wireframe.glb"),
            ImportItem(debug_models_path + "sphere_pink_wireframe.glb"),
            ImportItem(debug_models_path + "cube_gray_shaded.glb"),
            ImportItem(debug_models_path + "cube_green.glb"),
            ImportItem(debug_models_path + "cube_orange.glb"),
            ImportItem(debug_models_path + "cube_blue.glb"),
            ImportItem(debug_models_path + "cube_pink.glb"),
            ImportItem(debug_models_path + "cube_green_wireframe.glb"),
            ImportItem(debug_models_path + "cube_orange_wireframe.glb"),
            ImportItem(debug_models_path + "cube_blue_wireframe.glb"),
            ImportItem(debug_models_path + "cube_pink_wireframe.glb")
        ])

    if include_google_scanned:
        check_data_path(google_scanned_path, "google_scanned_path")
        # Instructions for Google Scanned objects:
        # 1. Find individual objects from https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
        # 2. Download object folder to e.g. source_data/google_scanned/Crosley_Alarm_Clock_Vintage_Metal/
        # 3. Manually move the texture image from e.g. Crosley_Alarm_Clock_Vintage_Metal/materials/textures/ to Crosley_Alarm_Clock_Vintage_Metal/meshes
        # 4. Add the item here following these examples.
        # 5. Run this script, check script terminal output, and copy-paste collisionBox 
        #    JSON entries to e.g. stretch_google_scanned.collection.json.
        # 6. Add object names to EpisodeGenerator.cpp generateBenchmarkEpisodeSet.
        import_items.extend([
            ImportItem(google_scanned_path + "Vtech_Stack_Sing_Rings_636_Months/meshes/model.obj", 
            name="Vtech_Stack_Sing_Rings_636_Months", decimate_fraction=0.1),
            ImportItem(google_scanned_path + "Crosley_Alarm_Clock_Vintage_Metal/meshes/model.obj", 
            name="Crosley_Alarm_Clock_Vintage_Metal", decimate_fraction=0.1),
            ImportItem(google_scanned_path + "Aroma_Stainless_Steel_Milk_Frother_2_Cup/meshes/model.obj", 
            name="Aroma_Stainless_Steel_Milk_Frother_2_Cup", decimate_fraction=0.1),
            ImportItem(google_scanned_path + "ACE_Coffee_Mug_Kristen_16_oz_cup/meshes/model.obj", 
            name="ACE_Coffee_Mug_Kristen_16_oz_cup", decimate_fraction=0.1)
        ])

    if include_ycb_objects:
        check_data_path(ycb_dataset_path, "ycb_dataset_path")
        import_items.extend([
            # ycb 16k meshes must be named and must be decimated (fraction was tuned by hand)
            ImportItem(ycb_dataset_path + "/ycb/024_bowl/google_16k/textured.obj", 
                name="024_bowl", decimate_fraction=0.1),
            # use source YCB cracker box with full-rez texture?
            # ImportItem("/home/eundersander/Downloads/003_cracker_box_google_16k (1)/003_cracker_box/google_16k/textured.obj", 
            #     name="003_cracker_box", decimate_fraction=0.05),
            ImportItem(ycb_dataset_path + "/ycb/003_cracker_box/google_16k/textured.obj", 
                name="003_cracker_box", decimate_fraction=0.05),
            ImportItem(ycb_dataset_path + "/ycb/010_potted_meat_can/google_16k/textured.obj", 
                name="010_potted_meat_can", decimate_fraction=0.05),
            ImportItem(ycb_dataset_path + "/ycb/002_master_chef_can/google_16k/textured.obj", 
                name="002_master_chef_can", decimate_fraction=0.2),
            ImportItem(ycb_dataset_path + "/ycb/004_sugar_box/google_16k/textured.obj", 
                name="004_sugar_box", decimate_fraction=0.05),
            ImportItem(ycb_dataset_path + "/ycb/005_tomato_soup_can/google_16k/textured.obj", 
                name="005_tomato_soup_can", decimate_fraction=0.05),
            ImportItem(ycb_dataset_path + "/ycb/009_gelatin_box/google_16k/textured.obj", 
                name="009_gelatin_box", decimate_fraction=0.05),
            ImportItem(ycb_dataset_path + "/ycb/008_pudding_box/google_16k/textured.obj", 
                name="008_pudding_box", decimate_fraction=0.05),
            ImportItem(ycb_dataset_path + "/ycb/007_tuna_fish_can/google_16k/textured.obj", 
                name="007_tuna_fish_can", decimate_fraction=0.05)
        ])

    if include_replicacad_stages:
        check_data_path(replicacad_stages_path, "replicacad_stages_path")

        glob_query = replicacad_stages_path + "*.glb"
        stage_filepaths = glob.glob(glob_query)
        if len(stage_filepaths) == 0:
            add_warning("No stage GLB files found matching '{}' .".format(glob_query))
        for filepath in stage_filepaths:
            import_items.extend([
                # for replicaCad_baked_lighting stages, I found vertex_merge_distance was better for mesh simplification compared to decimate.
                ImportItem(filepath, vertex_merge_distance=0.01),
            ])

    if include_mp3d_example:
        check_data_path(mp3d_example_path, "mp3d_example_path")
        import_items.extend([
            ImportItem(mp3d_example_path + "17DRP5sb8fy/17DRP5sb8fy.glb", do_join_all=False, 
                do_generate_gfx_replay=True,
                rotation=(1.570796, 'X'))
        ])


    def import_scene_helper(filepath):

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".glb" or ext == ".gltf":
            filename = os.path.basename(item.filepath)
            bpy.ops.import_scene.gltf(filepath=item.filepath, files=[{"name":filename, "name":filename}], loglevel=50)
        elif ext == ".obj":
            bpy.ops.import_scene.obj(filepath=item.filepath)
        elif ext == ".stl":
            bpy.ops.wm.stl_import(filepath=item.filepath)
        elif ext == ".dae":
            bpy.ops.wm.collada_import(filepath=item.filepath)
        else:
            raise RuntimeError("no importer found for " + filepath)

    for item in import_items:

        import_scene_helper(item.filepath)

        bpy.ops.object.select_all(action='SELECT')
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

        childless_empties = [e for e in bpy.context.selected_objects
                if e.type.startswith('EMPTY') and not e.children]
        if len(childless_empties):
            print("removing {} childless EMPTY nodes".format(len(childless_empties)))
            while childless_empties:
                bpy.data.objects.remove(childless_empties.pop())
            bpy.ops.object.select_all(action='SELECT')
            bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

        if item.do_join_all:
            if len(bpy.context.selected_objects) > 1:
                bpy.ops.object.join()
                bpy.ops.object.select_all(action='SELECT')
            o = bpy.context.selected_objects[0]
            bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            if item.name:
                joined_name = item.name
            else:
                filename = os.path.basename(item.filepath)
                joined_name = os.path.splitext(filename)[0]
            o.name = joined_name
            o.data.name = joined_name

            # print local bounding box
            min_corner = [o.bound_box[0][0], o.bound_box[0][1], o.bound_box[0][2]]
            max_corner = [o.bound_box[0][0], o.bound_box[0][1], o.bound_box[0][2]]
            for corner in range(8):
                for dim in range(3):
                    min_corner[dim] = min(min_corner[dim], o.bound_box[corner][dim])
                    max_corner[dim] = max(max_corner[dim], o.bound_box[corner][dim])
            # print(o.name, " min_corner: ", min_corner, ", max_corner: ", max_corner)
            # print("addFreeObject(set, \"" + o.name + "\", Mn::Range3D({" +
            #     str(min_corner[0]) + ", " + str(min_corner[2]) + ", " + str(-max_corner[1]) + 
            #     "}, {" +
            #     str(max_corner[0]) + ", " + str(max_corner[2]) + ", " + str(-min_corner[1]) +
            #     "}), sceneMapping);")
            print('"name": "' + o.name + '",')
            print('"collisionBox": {"min": [' 
                + str(min_corner[0]) + ', ' + str(min_corner[2]) + ', ' + str(-max_corner[1])
                + '], "max": ['
                + str(max_corner[0]) + ", " + str(max_corner[2]) + ", " + str(-min_corner[1])
                + ']},')
            print("")

            # remove extra materials because Bps only supports one material per mesh
            o = bpy.context.object
            if len(o.material_slots) > 1:
                bpy.ops.object.material_slot_remove_unused()
                if len(o.material_slots) > 1:
                    add_warning("{}: removing {} of {} materials! To avoid artifacts, you should refactor this source asset to only use one material.".format(
                        item.filepath, len(o.material_slots) - 1, len(o.material_slots)))
                    while len(o.material_slots) > 1:
                        o.active_material_index = 1
                        bpy.ops.object.material_slot_remove()

        if item.vertex_merge_distance > 0:
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.remove_doubles(threshold=item.vertex_merge_distance)
            bpy.ops.object.editmode_toggle()

        if item.decimate_fraction < 1.0:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.decimate(ratio=item.decimate_fraction, vertex_group_factor=1)
            bpy.ops.object.editmode_toggle()

        vertex_count = 0
        for o in bpy.context.selected_objects:
            vertex_count += len(o.data.vertices)
        threshold = 10000
        if vertex_count > threshold:
            add_warning("{}: vertex count of {} exceeds warning threshold of {}".format(
                item.filepath, vertex_count, threshold))

        # If no material, add a default maerial. This avoids later problems with GLTF
        # processing.
        if len(o.data.materials) == 0:
            mtrl = bpy.data.materials.new(name="default_material")
            mtrl.use_nodes = True
            o.data.materials.append(mtrl)

        if item.force_color:
            o = bpy.context.selected_objects[0]
            mtrl = o.data.materials[0]
            mtrl.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (item.force_color[0], item.force_color[1], item.force_color[2], 1)

        # temp hack: smooth shade everything for now
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.faces_shade_smooth()
        bpy.ops.object.editmode_toggle()

        if item.rotation:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.transform.rotate(value=item.rotation[0], orient_axis=item.rotation[1], orient_type='GLOBAL')
                # orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                # orient_matrix_type='GLOBAL', 
                # constraint_axis=(True, False, False), 
                # mirror=False, use_proportional_edit=False, 
                # proportional_edit_falloff='SMOOTH', 
                # proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)


        # Generate replay file for this ImportItem, so that we can use it as a static
        # scene in EpisodeGenerator.cpp, e.g. 17DRP5sb8fy.replay.json and 
        # addStaticScene(set, "17DRP5sb8fy").
        # Two ways of generating myscene.replay.json files:
        # 1. Here in Blender, for complex GLTF files that contain multiple meshes and
        #    materials and can't be merged using do_join_all. E.g. MP3D scene.
        # 2. Habitat-sim viewer --gala-write-scene-gfx-replay, for Habitat scenes
        #    comprised of a stage and objects, e.g. ReplicaCAD scene. Note
        #    each render asset in the scene must be imported here using do_join_all.
        # It isn't possible to combine #1 and #2 at this time, so e.g. we can't process
        # a Habitat scene containing an MP3D stage and additional objects.
        if item.do_generate_gfx_replay:
            
            keyframe_dict = {}
            keyframe_dict["creations"] = []
            keyframe_dict["stateUpdates"] = []

            instance_key = 0
            for o in bpy.context.selected_objects:

                # skip empty nodes
                if not len(o.data.polygons):
                    continue

                # make mesh name match object name
                o.data.name = o.name
                creation_dict = {}
                creation_dict["filepath"] = o.name
                creation_dict["isRGBD"] = True

                key_creation_dict = {}
                key_creation_dict["instanceKey"] = instance_key
                key_creation_dict["creation"] = creation_dict

                keyframe_dict["creations"].append(key_creation_dict)

                q = bpy.context.object.rotation_quaternion
                transform_dict = {}
                # note coordinate convention change
                transform_dict["rotation"] = [q[0], q[1], q[3], -q[2]]

                state_dict = {}
                state_dict["absTransform"] = transform_dict

                key_state_dict = {}
                key_state_dict["instanceKey"] = instance_key
                key_state_dict["state"] = state_dict
                # todo: include state here, including transform (not needed for MP3D scenes)

                keyframe_dict["stateUpdates"].append(key_state_dict)

                instance_key += 1

            replay_dict = {}
            replay_dict["keyframes"] = [keyframe_dict]

            # Serializing json
            import json
            json_object = json.dumps(replay_dict, indent=2)
            
            filepath = "./data/replays/" + os.path.splitext(os.path.basename(item.filepath))[0] + ".replay.json"
            print("Wrote ", filepath)
            with open(filepath, "w") as outfile:
                outfile.write(json_object)

                
                

        for o in bpy.context.selected_objects:
            o.hide_set(True)

    for o in bpy.context.scene.objects:
        o.hide_set(False)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_composite_glb_filepath = output_path + output_composite_glb_filename
    # bpy.ops.export_scene.gltf(filepath=output_composite_glb_filepath, check_existing=True, export_format='GLB')
    export_format = "GLTF_SEPARATE" if output_composite_glb_filename[:-5] == '.gltf' else "GLB"
    bpy.ops.export_scene.gltf(filepath=output_composite_glb_filepath, check_existing=True, export_format=export_format)
    print("")
    if len(warnings):
        print("{} warning{}:".format(len(warnings), "s" if len(warnings) > 1 else ""))
        for warning_str in warnings:
            print("  " + warning_str)
        print("")
    print("Success! {} render assets were exported to {}.".format(len(import_items), output_composite_glb_filepath))
    print("")
    result = input("Do you want to inspect the import results in the Blender GUI window? (y|n)")
    if result == 'n':
        exit(0)
    else:
        print("Close the Blender GUI window when you're finished inspecting.")

if __name__ == "__main__":
    main()
