#  simpleRT_plugin.py
#
#  Blender add-on for simpleRT render engine
#  a minimal ray tracing engine for CS148 HW5
#
#  Adding area light, sampling, and indirect illumination (GI)


bl_info = {
    "name": "simple_ray_tracer",
    "description": "Simple Ray-tracer for CS 148",
    "author": "CS148",
    "version": (0, 0, 2022),
    "blender": (3, 3, 0),
    "wiki_url": "http://web.stanford.edu/class/cs148/",
    "category": "Render",
}


import bpy
import numpy as np
from mathutils import Vector, Matrix
from math import sqrt, pi, cos, sin


def ray_cast(scene, origin, direction):
    return scene.ray_cast(scene.view_layers[0].depsgraph, origin, direction)


def RT_trace_ray(scene, ray_orig, ray_dir, lights, depth=0):
    # First, we cast a ray into the scene using Blender's built-in function
    has_hit, hit_loc, hit_norm, _, hit_obj, _ = ray_cast(scene, ray_orig, ray_dir)
    # set initial color (black) for the pixel
    color = np.zeros(3)
    # if the ray hits nothing in the scene, return black
    if not has_hit:
        return color
    # small offset to prevent self-occlusion for secondary rays
    eps = 1e-3
    # fix normal direction
    ray_inside_object = False
    if hit_norm.dot(ray_dir) > 0:
        hit_norm = -hit_norm
        ray_inside_object = True

    # get the ambient color of the scene
    ambient_color = scene.simpleRT.ambient_color
    # get the material of the object we hit
    mat = hit_obj.simpleRT_material
    # extract properties from the material
    diffuse_color = Vector(mat.diffuse_color).xyz
    specular_color = Vector(mat.specular_color).xyz
    specular_hardness = mat.specular_hardness

    # set flag for light hit. Will later be used to apply ambient light
    no_light_hit = True

    # iterate through all the lights in the scene
    for light in lights:
        # get light color
        light_color = np.array(light.data.color * light.data.energy / 4 / pi)
        light_loc = light.location
        # ----------
        # ADD CODE FOR AREA LIGHT HERE
        if light.data.type == "AREA":
            # calculate the normal vector for the area light
            light_normal = Vector((0, 0, -1))
            # rotate into the global world space
            light_normal.rotate(light.rotation_euler)

            # update the light color based on the tilt angle between the area light and object
            dir_light_to_hit = (hit_loc - light_loc).normalized()
            light_angle = light_normal.normalized().dot(dir_light_to_hit)
            light_color *= max(light_angle, 0)

            # calculate the point on the area light disk
            r = np.random.rand()
            theta = np.random.rand() * (pi * 2)
            x = sqrt(r) * cos(theta)
            y = sqrt(r) * sin(theta)
            radius = light.data.size / 2
            x *= radius
            y *= radius

            # transform it into the global world space
            light_loc_local = Vector((x, y, 0))
            light_loc = light.matrix_world @ light_loc_local
        # ----------
        # calculate vectors for shadow ray
        light_vec = light_loc - hit_loc
        light_dir = light_vec.normalized()
        new_orig = hit_loc + hit_norm * eps
        # cast shadow ray
        has_light_hit, light_hit_loc, _, _, _, _ = ray_cast(scene, new_orig, light_dir)
        if has_light_hit and (light_hit_loc - new_orig).length < light_vec.length:
            continue
        # Blinn-Phong diffuse
        I_light = light_color / light_vec.length_squared
        color += np.array(diffuse_color) * I_light * hit_norm.dot(light_dir)
        # Blinn-Phong specular
        half_vector = (light_dir - ray_dir).normalized()
        specular_reflection = hit_norm.dot(half_vector) ** specular_hardness
        color += np.array(specular_color) * I_light * specular_reflection
        # flag for ambient
        no_light_hit = False

    # Indirect Diffuse
    if depth > 0:
        # ----------------
        # establish a local coordinate system 
        # 
        # initial x guess
        z = hit_norm
        x = Vector((0, 0, 1))
        # if x too close to Z, start with a different guess
        xdotz = x.dot(z)
        if xdotz > 0.9 or xdotz < -0.9:
            x = Vector((0, 1, 0))
        # Gram Schmit orthogonalization
        x = x - (x.dot(z) * z)
        x = x.normalized()
        y = z.cross(x)
        # -----------------
        # sample a hemisphere oriented at (0, 0, 1)
        #
        r1 = np.random.rand()
        r2 = np.random.rand()
        cos_theta = r1
        sin_theta = sqrt(1 - cos_theta * cos_theta)
        phi = r2 * 2 * pi
        random_point = Vector((sin_theta * cos_theta, sin_theta * sin(phi), cos_theta))
        # ------------------
        # transfer ray direction into world space
        mat_transform = Matrix()
        mat_transform[0][0:3] = x
        mat_transform[1][0:3] = y
        mat_transform[2][0:3] = z
        mat_transform.transpose()
        global_ray_dir = mat_transform @ random_point
        # -------------------
        # recursively trace the ray
        raw_intensity = RT_trace_ray(scene, hit_loc + hit_norm * eps, global_ray_dir, lights, depth - 1)
        indirect_diffuse_color = raw_intensity * r1 * diffuse_color
        color += indirect_diffuse_color

    # ambient
    if no_light_hit:
        color += np.array(diffuse_color) * ambient_color

    # calculate reflectivity/fresnel
    reflectivity = mat.mirror_reflectivity
    if mat.use_fresnel:
        n2 = mat.ior
        r0 = ((1 - n2) / (1 + n2)) ** 2
        reflectivity = r0 + (1 - r0) * ((1 + ray_dir.dot(hit_norm)) ** 5)

    # recursive call for reflection and transmission
    if depth > 0:
        # reflection
        reflection_dir = (ray_dir - 2 * hit_norm * ray_dir.dot(hit_norm)).normalized()
        reflect_color = RT_trace_ray(
            scene, hit_loc + hit_norm * eps, reflection_dir, lights, depth - 1
        )
        color += reflectivity * reflect_color
        # transmission
        if mat.transmission > 0:
            if ray_inside_object:
                ior_ratio = mat.ior / 1
            else:
                ior_ratio = 1 / mat.ior
            under_sqrt = 1 - ior_ratio ** 2 * (1 - (ray_dir.dot(-hit_norm)) ** 2)
            if under_sqrt > 0:
                transmission_dir = ior_ratio * (
                    ray_dir - ray_dir.dot(hit_norm) * hit_norm
                ) - hit_norm * sqrt(under_sqrt)
                transmission_color = RT_trace_ray(
                    scene,
                    hit_loc - hit_norm * eps,
                    transmission_dir,
                    lights,
                    depth - 1,
                )
                color += (1 - reflectivity) * mat.transmission * transmission_color
    return color


# low-discrepancy sequence Van der Corput
def corput(n, base=2):
    q, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        q += remainder / denom
    return q - 0.5


def RT_render_scene(scene, width, height, depth, samples, buf):
    # get all lights from the scene
    scene_lights = [o for o in scene.objects if o.type == "LIGHT"]

    # get the location and orientation of the active camera
    cam_location = scene.camera.location
    cam_orientation = scene.camera.rotation_euler

    # get camera focal length
    focal_length = scene.camera.data.lens / scene.camera.data.sensor_width
    aspect_ratio = height / width

    sumbuf = np.zeros((height, width, 3))

    # sample different starting ray locations in the pixel area for each pixel
    dx = 1 / width
    dy = aspect_ratio / height
    corput_x = [corput(i, 2) * dx for i in range(samples)]
    corput_y = [corput(i, 3) * dy for i in range(samples)]

    # iterate on samples
    for s in range(samples):
        # iterate through all the pixels, cast a ray for each pixel
        for y in range(height):
            # get screen space coordinate for y
            screen_y = ((y - (height / 2)) / height) * aspect_ratio
            for x in range(width):
                # get screen space coordinate for x
                screen_x = (x - (width / 2)) / width

                ray_dir = Vector(
                    (screen_x + corput_x[s], screen_y + corput_y[s], -focal_length)
                )

                ray_dir.rotate(cam_orientation)
                ray_dir = ray_dir.normalized()

                sumbuf[y, x, 0:3] += RT_trace_ray(
                    scene, cam_location, ray_dir, scene_lights, depth
                )

                buf[y, x, 0:3] = sumbuf[y, x] / (samples + 1)

                # populate the alpha component of the buffer
                # to make the pixel not transparent
                buf[y, x, 3] = 1
            yield y + s * height
    return buf


# modified from https://docs.blender.org/api/current/bpy.types.RenderEngine.html
class SimpleRTRenderEngine(bpy.types.RenderEngine):
    bl_idname = "simple_RT"
    bl_label = "SimpleRT"
    bl_use_preview = False

    def __init__(self):
        self.draw_data = None

    def __del__(self):
        pass

    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)

        # ----------
        self.samples = depsgraph.scene.simpleRT.samples
        # ----------

        if self.is_preview:
            pass
        else:
            self.render_scene(scene)

    def render_scene(self, scene):
        height, width = self.size_y, self.size_x
        buf = np.zeros((height, width, 4))

        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]

        # get the maximum ray tracing recursion depth
        depth = scene.simpleRT.recursion_depth

        samples = self.samples
        total_height = samples * height

        # time the render
        import time
        from datetime import timedelta

        start_time = time.time()

        # start ray tracing
        update_cycle = int(10000 / width)
        for y in RT_render_scene(scene, width, height, depth, samples, buf):

            elapsed = int(time.time() - start_time)
            remain = int(elapsed / (y + 1) * (total_height - y - 1))
            status = (
                f"pass {y//height+1}/{samples} "
                + f"| Remaining {timedelta(seconds=remain)}"
            )
            self.update_stats("", status)
            print(status, end="\r")
            # update Blender progress bar
            self.update_progress(y / total_height)
            # update render result
            # update too frequently will significantly slow down the rendering
            if y % update_cycle == 0 or y == total_height - 1:
                self.update_result(result)
                layer.rect = buf.reshape(-1, 4).tolist()

            # catch "ESC" event to cancel the render
            if self.test_break():
                break

        # tell Blender all pixels have been set and are final
        self.end_result(result)


def register():
    bpy.utils.register_class(SimpleRTRenderEngine)


def unregister():
    bpy.utils.unregister_class(SimpleRTRenderEngine)


if __name__ == "__main__":
    register()